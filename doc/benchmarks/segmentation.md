
# Deep Globe Road Extraction

Ours

[Val] OA (micro): accuracy: 0.9100 - precision: 0.6501 - recall: 0.9438 - kappa: 0.4262 - mean_iou: 0.6017 - f1_score: 0.7058


# Flood3i (few-shots v.s. fine-tune)


full finetuned

[Val] OA (micro): accuracy: 0.9131 - precision: 0.8519 - recall: 0.8470 - kappa: 0.8656 - mean_iou: 0.6557 - f1_score: 0.8487

few-shots: 5 shot

class1
[Val] OA (micro): accuracy: 0.8481 - precision: 0.5483 - recall: 0.8934 - kappa: 0.1504 - mean_iou: 0.4918 - f1_score: 0.546


# Atlantic Forest
<!--
[Val] OA (micro): accuracy: 0.9756 - precision: 0.9732 - recall: 0.9724 - kappa: 0.9456 - mean_iou: 0.9302 - f1_score: 0.9728
 -->

[Val] OA (micro): accuracy: 0.9732 - precision: 0.9697 - recall: 0.9704 - kappa: 0.9401 - f1: [0.9797119498252869, 0.9603999257087708] - dice: 0.9445 - miou: 0.9239 - jaccard: [0.9602307677268982, 0.9238167405128479] - loss: 2.462e-01
23:28:57 - ℹ  [ INFO ] hyper_latent_segmentation_trainer.py:1178 -
---------------- Validation infomation ----------------
{'val/accuracy': 0.9731696844100952, 'val/precision': 0.969668984413147, 'val/recall': 0.9704464673995972, 'val/kappa': 0.9401119947433472, 'val/f1': [0.9797119498252869, 0.9603999257087708], 'val/dice': 0.9444625377655029, 'val/miou': 0.9239232540130615, 'val/jaccard': [0.9602307677268982, 0.9238167405128479], 'val/dice_macro': 0.9586333632469177, 'val/dice_per_class': [0.9767417311668396, 0.9405249953269958], 'val/miou_macro': 0.9239233434200287, 'val/miou_per_class': [0.9549543261528015, 0.8928923606872559]}


# SOS Oil leakage


<!-- -------- old metrics

20:00:01 - ℹ  [ INFO ] hyper_latent_segmentation_trainer.py:359 - [Val] OA (micro): accuracy: 0.9307 - precision: 0.9251 - recall: 0.9232 - kappa: 0.8483 - mean_iou: 0.7890 - f1_score: 0.9241 - loss: 6.252e-01
{'val/accuracy': 0.930685818195343, 'val/precision': 0.9251285195350647, 'val/recall': 0.9231830835342407, 'val/kappa': 0.8482835292816162, 'val/mean_iou': 0.7890297174453735, 'val/f1_score': 0.9241406917572021, 'val/dice_macro': 0.844771146774292, 'val/dice_per_class': [0.8878612518310547, 0.8016810417175293], 'val/mean_iou_macro': 0.7883211374282837, 'val/mean_iou_per_class': [0.8412116765975952, 0.7354305982589722]}


---- new -->


[Val] OA (micro): accuracy: 0.9296 - precision: 0.9273 - recall: 0.9180 - kappa: 0.8447 - f1: [0.946148693561554, 0.8985452651977539] - dice: 0.8002 - miou: 0.7860 - jaccard: [0.8978009819984436, 0.8157804608345032] - loss: 6.418e-01
23:22:19 - ℹ  [ INFO ] hyper_latent_segmentation_trainer.py:1178 -
---------------- Validation infomation ----------------
{'val/accuracy': 0.9296425580978394, 'val/precision': 0.92734694480896, 'val/recall': 0.9179947376251221, 'val/kappa': 0.8447397947311401, 'val/f1': [0.946148693561554, 0.8985452651977539], 'val/dice': 0.8002251386642456, 'val/miou': 0.7860118746757507, 'val/jaccard': [0.8978009819984436, 0.8157804608345032], 'val/dice_macro': 0.8432268500328064, 'val/dice_per_class': [0.8872584700584412, 0.7991952300071716], 'val/miou_macro': 0.7852773368358612, 'val/miou_per_class': [0.8401603698730469, 0.7303943037986755]}


# c2smsflood

 [Val] OA (micro): accuracy: 0.9678 - precision: 0.9291 - recall: 0.9392 - kappa: 0.8682 - f1: [0.981217622756958, 0.8869331479072571] - dice: 0.5837 - miou: 0.6970 - jaccard: [0.9631277322769165, 0.7968372106552124] - loss: 5.238e-01
23:04:02 - ℹ  [ INFO ] hyper_latent_segmentation_trainer.py:1178 -
---------------- Validation infomation ----------------
{'val/accuracy': 0.967786431312561, 'val/precision': 0.9290914535522461, 'val/recall': 0.9392387866973877, 'val/kappa': 0.8681550025939941, 'val/f1': [0.981217622756958, 0.8869331479072571], 'val/dice': 0.5836670398712158, 'val/miou': 0.697048544883728, 'val/jaccard': [0.9631277322769165, 0.7968372106552124], 'val/dice_macro': 0.7122762203216553, 'val/dice_per_class': [0.9348227381706238, 0.48972970247268677], 'val/miou_macro': 0.6883613765239716, 'val/miou_per_class': [0.9344989061355591, 0.44222384691238403]}
