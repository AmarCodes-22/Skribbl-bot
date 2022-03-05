# Skribbl-bot
This project uses a simple ResNet model to identify doodle drawings in online game [Skribbl.io](https://skribbl.io/)

The model used is trained on [Google's Quickdraw Dataset](https://quickdraw.withgoogle.com/data)

This project uses [OpenCV](https://opencv.org/) to capture the drawings being drawn by the player, [PyTorch](https://pytorch.org/) as the deep learning framework, and [Torchserve](https://pytorch.org/serve/#:~:text=TorchServe%20is%20a%20performant%2C%20flexible,eager%20mode%20and%20torschripted%20models.) to deploy model on localhost.

Some of the outputs of the model being predicted in real-time are shown below:-
Don't focus on the word shown by Skribbl, didn't use custom words for the game

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/airplane.png)

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/bear.png)

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/car.png)

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/cat.png)

![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/cruise%20ship.png)

Please don't judge me 
![](https://github.com/AmarCodes-22/Skribbl-bot/blob/main/assets/predictions/horse.png)

There are no steps to reproduce, however if you want to try it out on your own system, feel free to get in touch.
