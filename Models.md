Baselines

Unimodal
- [CNN](models/baselines/mulsa/lightning_logs/sorting_imi_vg_simple_seqlen_1_spec04-22-07:46:29)
- [CNN+LSTM](models/baselines/mulsa/lightning_logs/sorting_imi_vg_lstm_seqlen_3_spec04-22-00:48:38)
- [MHA](models/baselines/mulsa/lightning_logs/sorting_imi_vg_simple_seqlen_1_mha_spec04-22-04:18:40)

Multimodal
- [CNN](models/baselines/mulsa/lightning_logs/sorting_imi_vg_ag_simple_seqlen_1_spec04-22-17:19:32)
- [CNN+LSTM](models/baselines/mulsa/lightning_logs/sorting_imi_vg_ag_lstm_seqlen_3_spec04-22-21:39:20)
- [MHA](models/baselines/mulsa/lightning_logs/sorting_imi_vg_ag_simple_seqlen_1_mha_spec04-22-15:08:58)


Proposed

- [MHA+LSTM](models/baselines/mulsa/lightning_logs/sorting_imi_vg_ag_lstm_seqlen_3_mha_spec04-22-19:26:28)
- [MHA+LSTM Unimodal (not used)](models/baselines/mulsa/lightning_logs/sorting_imi_vg_lstm_seqlen_3_mha_spec04-21-21:13:43)

Flags

- seq_len
- use_mha
- use_audio

|            | Audio | LSTM (seq_len>1) | MHA |
|------------|:-----:|:----------------:|:---:|
| CNN        |       |                  |     |
| CNN + LSTM |       |         Y        |     |
| MHA        |       |                  |  Y  |
| CNN        |   Y   |                  |     |
| CNN + LSTM |   Y   |         Y        |     |
| MHA        |   Y   |                  |  Y  |