#!/bin/bash

torchserve --ncs --start --model-store ./serve/baseline/model_store --models resnet-18=resnet-18.mar
