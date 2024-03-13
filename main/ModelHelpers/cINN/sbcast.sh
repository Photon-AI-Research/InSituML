#! /bin/bash

sbcast -pf ./tml3.tar.gz /mnt/bb/$USER/tml3.tar.gz

mkdir /mnt/bb/$USER/tml3/

tar -xzf /mnt/bb/$USER/tml3.tar.gz -C /mnt/bb/$USER/tml3/

source /mnt/bb/vineethg/tml3/bin/activate