import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def main(
    name: Annotated[str, typer.Argument(help="Name of the model to test")],
    age: Annotated[int, typer.Argument(help="Age of the model to test")],
):
    print(f"{name} has {age} years of experience in testing models.")


if __name__ == "__main__":
    app()
