##### output sequence length
# python3 train.py --mha --decoder "lstm" --use_audio  --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --output_sequence_length 5  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --output_sequence_length 7  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --output_sequence_length 9  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"


# # ##### audio length
# python3 train.py --mha --decoder "lstm" --use_audio --audio_len 1 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --audio_len 3 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --audio_len 5 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
python3 train.py --mha --decoder "lstm" --use_audio --audio_len 7 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"


# ##### num stacks
python3 train.py --mha --decoder "lstm" --use_audio --num_stack 3 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"
# python3 train.py --mha --decoder "lstm" --use_audio --num_stack 9 --output_sequence_length 3  --config_path "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/new/train.yaml"

### lstm hidden layers, headstr