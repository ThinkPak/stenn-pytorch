from src.backbones import utae, unet3d, convlstm, convgru, vgg, stenn, stenn_nodense, stenn_notransformer


def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=config.input_channel,
                encoder_widths=[64,64,64,128],
                decoder_widths=[32,32,64,128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                agg_mode="att_group",
                encoder_norm="group",
                n_head=16,
                d_model=256,
                d_k=4,
                encoder=False,
                return_maps=False,
                pad_value=0,
                padding_mode="reflect",
            )
        elif config.model == "vgg":
            model = vgg.VGG(
                input_dim=config.input_channel,
                encoder_widths=[16, 32, 32, 64],
                out_widths=[64],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                pad_value=0,
                encoder_norm="group",
                padding_mode="reflect",
                num_heads=8,
                dim_feedforward=128,
                num_layers=1,
                num_classes=20,
                batch_row=1,
            )
        elif config.model == "stenn":
            model = stenn.STENN(
                input_dim=config.input_channel,
                encoder_widths=[16, 32, 32, 64],
                dilation_rates=[[1, 2, 3],
                                [2, 4, 6],
                                [4, 8, 12],
                                [8, 16, 24]],
                pad_value=0,
                encoder_norm="group",
                padding_mode="reflect",
                num_heads=8,
                dim_feedforward=256,
                num_layers=1,
                batch_row=1,
                out_widths=[64],
                num_classes=20,
            )
        elif config.model == "stenn_nodense":
            model = stenn_nodense.STENN_NoDense(
                input_dim=config.input_channel,
                encoder_widths=[16, 32, 32, 64],
                dilation_rates=[[1, 2, 3],  # 可以修改
                                [2, 4, 6],
                                [4, 8, 12],
                                [8, 16, 24]],
                pad_value=0,
                encoder_norm="group",
                padding_mode="reflect",
                num_heads=8,
                dim_feedforward=256,
                num_layers=1,  # 可以修改
                batch_row=1,  # 可以修改
                out_widths=[64],
                num_classes=20,
            )
        elif config.model == "stenn_notransformer":
            model = stenn_notransformer.STENN_NoTransformer(
                input_dim=config.input_channel,
                encoder_widths=[16, 32, 32, 64],
                dilation_rates=[[1, 2, 3],  # 可以修改
                                [2, 4, 6],
                                [4, 8, 12],
                                [8, 16, 24]],
                pad_value=0,
                encoder_norm="group",
                padding_mode="reflect",
                out_widths=[64],
                num_classes=20,
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif config.model == "uconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif config.model == "buconvlstm":
            model = utae.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
        else:
            raise NotImplementedError
        return model
    else:
        raise NotImplementedError
