"""Inference-only RTMPose configuration for the infant head_top checkpoint."""

default_scope = "mmpose"

dataset_meta = dict(
    dataset_name="infant_head_top",
    keypoint_info=dict(
        {
            0: dict(
                color=[51, 153, 255],
                name="head_top",
                swap="head_top",
                type="upper",
            )
        }
    ),
    skeleton_info=dict(),
    joint_weights=[1.0],
    sigmas=[0.05],
)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            checkpoint=(
                "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                "cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth"
            ),
            prefix="backbone.",
        ),
        out_indices=(4,),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=768,
        out_channels=1,
        input_size=(192, 256),
        in_featuremap_size=(6, 8),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=(4.9, 5.66),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False,
        ),
    ),
    test_cfg=dict(flip_test=True),
)

# mmpose.apis.inference_topdown reads this pipeline directly at inference time.
test_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type="LoadImage", backend_args=dict(backend="local")),
            dict(type="GetBBoxCenterScale"),
            dict(type="TopdownAffine", input_size=(192, 256)),
            dict(type="PackPoseInputs"),
        ]
    )
)
