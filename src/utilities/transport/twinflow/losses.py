import copy

import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Transport Solver (Mathematical Core)
# ---------------------------------------------------------
class LinearSchedule:
    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1


class TransportSolver(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.sched = LinearSchedule()

    def get_params(self, t):
        t = t.view(-1, 1, 1, 1)
        return (self.sched.alpha_in(t), self.sched.gamma_in(t), self.sched.alpha_to(t), self.sched.gamma_to(t))

    def sample_location(self, x1, x0, t):
        a_in, g_in, _, _ = self.get_params(t)
        return g_in * x0 + a_in * x1

    def get_target_velocity(self, x1, x0, t):
        _, _, a_to, g_to = self.get_params(t)
        return a_to * x1 + g_to * x0

    def predict_x0_from_velocity(self, x_t, t, F_t):
        """
        Solve for x0 (x_hat) and z (z_hat) from x_t and velocity F_t
        """
        a_in, g_in, a_to, g_to = self.get_params(t)
        dent = a_in * g_to - g_in * a_to

        # Numerical protection to avoid division by zero
        dent = torch.where(dent.abs() < 1e-5, -1.0 * torch.ones_like(dent), dent)

        x_hat = (F_t * a_in - x_t * a_to) / dent
        z_hat = (x_t * g_to - F_t * g_in) / dent
        return x_hat, z_hat


class BaseFlowLoss(nn.Module):
    """
    Base class: Provides common Transport Solver and utility functions
    """

    def __init__(self, transport_solver: TransportSolver):
        super().__init__()
        self.transport = transport_solver

    def get_velocity(self, model, x, t, y):
        """Common forward interface"""
        return model(x, t, y)

    def predict_x0(self, model, x_t, t, y):
        """Common x0 prediction interface"""
        v = self.get_velocity(model, x_t, t, y)
        x_hat, z_hat = self.transport.predict_x0_from_velocity(x_t, t, v)
        return x_hat, z_hat, v


class FlowMatchingLoss(BaseFlowLoss):
    """
    Implements N=0 Standard Flow Matching Loss.
    This is the most basic pre-training loss, and can also serve as TwinFlow's L_base (if not using N=2).

    Ref: Eq (1) with N=0 (Standard FM objective)
    Target is Ground Truth Velocity (z - x).
    """

    def __init__(self, transport_solver: TransportSolver):
        super().__init__(transport_solver)

    def compute_loss(self, model, x, y):
        batch_size = x.shape[0]
        device = x.device

        # 1. Sample time t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # 2. Generate noise z
        z = torch.randn_like(x)

        # 3. Interpolate to get x_t
        # x_t = (1-t)x + t*z
        x_t = self.transport.sample_location(z, x, t)

        # 4. Compute Ground Truth Velocity
        # target_v = z - x
        target_v = self.transport.get_target_velocity(z, x, t)

        # 5. Model prediction
        pred_v = self.get_velocity(model, x_t, t, y)

        # 6. MSE Loss
        loss = torch.mean((pred_v - target_v) ** 2)

        return loss


class RCGMLoss(BaseFlowLoss):
    """
    Implements N=2 RCGM (Recursive Consistent Generation Model) Loss.
    This is the L_base part of the TwinFlow paper.

    Ref: [cite: 101] Eq (1) with N=2
    """

    def __init__(self, transport_solver, teacher_model):
        super().__init__(transport_solver)
        # Teacher model for computing stable Target (sum f_theta- in Eq 1)
        # If no teacher provided, use student (self-target), but usually unstable
        self.teacher_model = teacher_model

    def heun_step(self, model, x, t, y, dt):
        """
        Use Heun Solver (2nd Order) for more precise Teacher sampling
        x_{t+1} = x_t + 0.5 * (v_t + v_{t+1_guess}) * dt
        """
        with torch.no_grad():
            # 1. Euler predict first step (Predictor)
            v_t = model(x, t, y)
            d_t = dt.view(-1, 1, 1, 1)
            x_guess = x + v_t * d_t

            # 2. Compute next time step
            t_next = t + dt

            # [Boundary protection] If t_next < 0 (because dt is negative), clamp
            # But here dt is step size, usually controlled externally

            # 3. Compute velocity at predicted point (Corrector)
            v_next = model(x_guess, t_next, y)

            # 4. Average velocity update
            v_avg = 0.5 * (v_t + v_next)
            x_next = x + v_avg * d_t

        return x_next, v_avg  # Return v_avg for displacement calculation

    def sample_any_n_trajectory(self, x_t, t, y, n_steps=2):
        """
        [Generic N-Step]
        Use Teacher to take N steps to estimate Target x0.
        """
        model_target = self.teacher_model

        # Divide the time interval from t to 0 into n_steps segments
        # dt is negative because time goes from t -> 0
        total_duration = 0 - t
        dt = total_duration / n_steps

        x_curr = x_t
        t_curr = t

        # Loop N-1 times for stepping
        for _ in range(n_steps - 1):
            x_curr = self.heun_step(model_target, x_curr, t_curr, y, dt)
            t_curr = t_curr + dt
            t_curr = torch.clamp(t_curr, min=1e-5)  # Prevent numerical overflow

        # Final step (Project): Direct prediction from final t_curr to 0
        # This is the definition of Consistency: directly shoot to x0 in the final step
        with torch.no_grad():
            v_end = model_target(x_curr, t_curr, y)
            d_end = (0 - t_curr).view(-1, 1, 1, 1)
            target_x0 = x_curr + v_end * d_end

        return target_x0

    def sample_n2_trajectory(self, x_t, t, y, dt):
        """
        N=2 trajectory sampling (using Heun for enhanced precision)
        t -> t1 -> t2 -> 0
        """
        # [Key point 2]: Always use Teacher (EMA) to compute Target
        model_target = self.teacher_model

        # Define time points
        # Note: In Flow Matching we go backward (1 -> 0), so dt should be negative
        # For code clarity, dt is passed as positive here, we negate it in computation
        step_size = -dt

        t1 = t + step_size
        t2 = t1 + step_size

        # Boundary handling, prevent time < 0
        t1 = torch.clamp(t1, min=1e-5)  # Keep a small epsilon to avoid numerical issues
        t2 = torch.clamp(t2, min=1e-5)

        # -------------------------------------------------------
        # Step 1: t -> t1 (using Heun)
        # -------------------------------------------------------
        # Note: dt here is step_size (negative)
        # Actually Heun needs t_next, we manually control it

        # Simplified Heun:
        x_t1, v_avg_1 = self.heun_step(model_target, x_t, t, y, step_size)

        # -------------------------------------------------------
        # Step 2: t1 -> t2 (using Heun)
        # -------------------------------------------------------
        # Update step_size because t2 might be clamped
        step_size_2 = t2 - t1
        x_t2, v_avg_2 = self.heun_step(model_target, x_t1, t1, y, step_size_2)

        # -------------------------------------------------------
        # Step 3: t2 -> 0 (final step usually uses Euler shooting, or Heun as well)
        # -------------------------------------------------------
        # Consistency is typically defined as: f(x_t2, 0)
        # That is, predict x0 at time t2. Just output v * (0 - t2)
        with torch.no_grad():
            v_end = model_target(x_t2, t2, y)
            d_end = (0 - t2).view(-1, 1, 1, 1)

            # Final Target x0
            # Path: x_t -> x_t1 -> x_t2 -> (predict) -> x0
            target_x0 = x_t2 + v_end * d_end

        return target_x0

    def compute_loss(self, student_model, x, y):
        self.student_model = student_model  # Hack for self-target if teacher is None

        batch_size = x.shape[0]
        device = x.device

        # 1. Sample t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # 2. Generate noise and interpolate to get x_t
        z = torch.randn_like(x)
        x_t = self.transport.sample_location(z, x, t)

        # 3. Define step size dt for N=2 discretization
        # Divide t to 0 into 3 segments (t->t1, t1->t2, t2->0) or 2 segments for N=2?
        # RCGM N=2 usually refers to recursive depth. Here we take a reasonable dt
        # For example, dt = t / 2.0, so t -> t/2 -> 0
        dt_step = t / 2.0

        # Get Target (using Heun + EMA)
        target_x0 = self.sample_n2_trajectory(x_t, t, y, dt_step)

        # Student prediction
        pred_x0, _, _ = self.predict_x0(student_model, x_t, t, y)

        loss = torch.mean((pred_x0 - target_x0.detach()) ** 2)
        return loss


class TwinFlowLoss(BaseFlowLoss):
    """
    Implements TwinFlow-specific L_adv and L_rectify.

    Ref: [cite: 130] Eq (2) for L_adv
    Ref: [cite: 166] Eq (9) for L_rectify
    """

    def __init__(self, transport_solver):
        super().__init__(transport_solver)

    def compute_loss(self, student_model, x, y):
        batch_size = x.shape[0]
        device = x.device

        # ---------------------------------------------------
        # Phase 1: Generate Fake Data (Generator Step)
        # ---------------------------------------------------
        t_one = torch.ones(batch_size, device=device)
        z_input = torch.randn_like(x)

        with torch.no_grad():
            # Predict x_fake using t=1 (One-step generation)
            # Note: No gradient needed here, as we train "how to fit this trajectory", not optimize generation itself
            # (Although needed in GAN, TwinFlow uses rectify logic)
            x_fake, _, _ = self.predict_x0(student_model, z_input, t_one, y)

        # ---------------------------------------------------
        # Phase 2: Build Fake Trajectory
        # ---------------------------------------------------
        z_fake_noise = torch.randn_like(x)
        t_prime = torch.rand(batch_size, device=device)  # t' in [0, 1]

        # Sample on Fake trajectory: connecting z_fake_noise -> x_fake
        x_t_fake = self.transport.sample_location(z_fake_noise, x_fake, t_prime)

        # ---------------------------------------------------
        # Phase 3: Self-Adversarial Loss (L_adv)
        # Ref: [cite: 130] L_adv = d(F(x_fake, -t'), z_fake - x_fake)
        # ---------------------------------------------------
        # Target: Velocity pointing from z_fake_noise to x_fake
        target_v_adv = self.transport.get_target_velocity(z_fake_noise, x_fake, t_prime)

        # Model Input: Negative Time (-t')
        pred_v_neg = self.get_velocity(student_model, x_t_fake, -t_prime, y)

        loss_adv = torch.mean((pred_v_neg - target_v_adv) ** 2)

        # ---------------------------------------------------
        # Phase 4: Rectification Loss (L_rectify)
        # Ref: [cite: 166] L_rectify = d(F(x_fake, t'), sg[F(x_fake, -t')])
        # ---------------------------------------------------
        # Model Input: Positive Time (t')
        # This represents "Real Interpretation"
        pred_v_pos = self.get_velocity(student_model, x_t_fake, t_prime, y)

        # Target: Negative prediction (Detached/Stop-Gradient)
        # Since L_adv forces negative time prediction to point to x_fake, L_rectify forces positive time prediction to also point to x_fake
        # Thus straightening the trajectory.
        loss_rectify = torch.mean((pred_v_pos - pred_v_neg.detach()) ** 2)

        return loss_adv, loss_rectify


@torch.no_grad()
def sample_cfg_unified(model, device, labels, steps=1, cfg_scale=4.0):
    """
    labels: tensor of shape (N,) with values 0-9
    """
    model.eval()
    transport = TransportSolver(device)
    num_samples = labels.shape[0]
    z = torch.randn(num_samples, 1, 28, 28).to(device)

    # Create null labels for unconditional prediction
    null_token = 10
    y_null = torch.ones_like(labels) * null_token

    t_steps = torch.linspace(1, 0, steps + 1).to(device)
    x_curr = z

    for i in range(steps):
        t_curr = t_steps[i].repeat(num_samples)
        t_next = t_steps[i + 1].repeat(num_samples)

        # --- CFG Prediction ---
        # 1. Conditional Prediction
        v_cond = model(x_curr, t_curr, labels)
        # 2. Unconditional Prediction
        v_uncond = model(x_curr, t_curr, y_null)

        # 3. Guidance
        v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

        # --- Unified Sampler Step ---
        # 1. Solve for x_hat using Guided Velocity
        x_hat, z_hat = transport.predict_x0_from_velocity(x_curr, t_curr, v_pred)

        # 2. Mix to next step
        a_next, g_next, _, _ = transport.get_params(t_next)
        x_next = g_next * x_hat + a_next * z_hat

        x_curr = x_next

    return x_curr.cpu()


# ==============================================================================
# Helper: Integrate into training loop
# ==============================================================================


def train_distill_refactored(model, loader, device, epochs=10):
    print(f"\n>>> Starting TwinFlow Distillation (Refactored)...")

    # 1. Initialize Transport
    transport = TransportSolver(device)

    # 2. Initialize EMA Teacher (for RCGM N=2)
    # Must be a deep copy of model and not participate in gradient updates
    teacher_model = copy.deepcopy(model)
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    # 3. Initialize Loss Modules
    loss_rcgm = RCGMLoss(transport, teacher_model=teacher_model)
    loss_twin = TwinFlowLoss(transport)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)  # Slightly increased LR

    # EMA update rate
    ema_decay = 0.999

    for epoch in range(epochs):
        total_base = 0
        total_adv = 0
        total_rect = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Dropout logic externally applied or inside get_velocity if needed
            # Here assuming y is already dropped out or full

            optimizer.zero_grad()

            # A. Base Loss (RCGM N=2)
            #  "We adopt N=2 formulation of (1) to enhance the training stability."
            l_base = loss_rcgm.compute_loss(model, x, y)

            # B. TwinFlow Loss
            l_adv, l_rect = loss_twin.compute_loss(model, x, y)

            # Total Loss
            # Weight strategy here:
            # RCGM is responsible for stability, TwinFlow for straightening.
            # If straightening failed before, try reducing l_base weight or increase epochs
            loss = 1.0 * l_base + 1.0 * (l_adv + l_rect)

            loss.backward()
            optimizer.step()

            # Update Teacher (EMA)
            with torch.no_grad():
                for p_s, p_t in zip(model.parameters(), teacher_model.parameters()):
                    p_t.data.mul_(ema_decay).add_(p_s.data, alpha=1 - ema_decay)

            total_base += l_base.item()
            total_adv += l_adv.item()
            total_rect += l_rect.item()

        print(
            f"Ep {epoch + 1} | Base: {total_base / len(loader):.4f} | Adv: {total_adv / len(loader):.4f} | Rect: {total_rect / len(loader):.4f}"
        )
