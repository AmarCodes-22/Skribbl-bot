# Skribbl-bot
This project uses a simple ResNet model to identify doodle drawings in online game [Skribbl.io](https://skribbl.io/)

The model used is trained on [Google's Quickdraw Dataset](https://quickdraw.withgoogle.com/data)

This project uses [OpenCV](https://opencv.org/) to capture the drawings being drawn by the player, [PyTorch](https://pytorch.org/) as the deep learning framework, and [Torchserve](https://pytorch.org/serve/#:~:text=TorchServe%20is%20a%20performant%2C%20flexible,eager%20mode%20and%20torschripted%20models.) to deploy model on localhost.

Some of the outputs of the model being predicted in real-time are shown below:-

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/airplane.png)

