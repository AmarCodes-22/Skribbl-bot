echo '**********debug**********'
which apt-get

python src/scripts/install_dependencies.py

torchserve --start --model-store ./serve/baseline/model_store --models resnet-18=resnet-18.mar --ncs
