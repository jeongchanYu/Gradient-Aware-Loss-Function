##### CHECKPOINT_SETTING #######################################################################
save_checkpoint_name            = "Proposed_FD_GA-LSD1†_lr-3"
load_checkpoint_name            = "Proposed_FD_GA-LSD1†_lr-3_795"
save_checkpoint_period          = 1
save_optimizer_period           = 10
##### LEARNING_PARAMETER #######################################################################
epochs                          = 1000
batch_size                      = 16
frame_size                      = 16384 * 4
eps                             = 1e-12
learning_rate                   = 1e-3
seed                            = 42
##### MODEL_PARAMETER ##########################################################################
fft_size                        = 512
hidden_size                     = 256
##### DATASET_PATH #############################################################################
dataset_path                    = "~/DATASET/NSDTSEA_28spk_16k"
sample_rate                     = 16000