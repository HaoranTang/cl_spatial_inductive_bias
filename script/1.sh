# ./in100_mocov2b.sh ori
# ./in100_mocov2b.sh glo_2
# ./in100_mocov2b.sh loc_128
# ./in100_mocov2b.sh glo_4
# ./in100_mocov2b.sh loc_64
./in100_mocov2b.sh gam_0.2 --resume_from_checkpoint=./trained_models/mocov2b-resnet18-in100-200ep/gam_0.2/mocov2b-resnet18-in100-200ep-gam_0.2-ep=150_.ckpt
./in100_mocov2b.sh gam_5
./in100_mocov2b.sh glo_8
./in100_mocov2b.sh loc_32
