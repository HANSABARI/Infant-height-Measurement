import cv2
import numpy as np
import torch
import os
from mobile_sam import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 144 / 255, 30 / 255, 0.6])  # 파란색 반투명 마스크
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def main():
    # Mac 환경 테스트
    device = "cpu"
    print(f"Using device: {device}")

    # MobileSAM 로드
    model_type = "vit_t"
    sam_checkpoint = "weights/mobile_sam.pt"

    if not os.path.exists(sam_checkpoint):
        print("가중치 파일이 없습니다. 경로를 확인해주세요.")
        return

    print("MobileSAM 모델을 불러오는 중...")
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    # 테스트 이미지 로드
    image_path = "/Users/ryuyoonmin/PycharmProjects/h-align-server/debug_images/test_body.jpeg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # RGB Format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이후 RTMDet에서 BBox를 넘겨줄 예정
    h, w, _ = image.shape
    input_box = np.array([w * 0.7, h * 0.35, w * 0.83, h * 0.55])

    # MobileSAM 추론
    print("마스크 추론 중...")
    predictor.set_image(image_rgb)
    masks, scores, logits = predictor.predict(
        box=input_box,
        multimask_output=False  # 가장 신뢰도 높은 1개 마스크만 출력
    )

    # 결과 시각화
    mask = masks[0]
    score = scores[0]
    color_mask = show_mask(mask, None)
    overlay_image = image.copy()

    for c in range(3):
        overlay_image[:, :, c] = np.where(mask == 1,
                                          overlay_image[:, :, c] * 0.5 + color_mask[:, :, c] * 255 * 0.5,
                                          overlay_image[:, :, c])

    cv2.rectangle(overlay_image, (int(input_box[0]), int(input_box[1])), (int(input_box[2]), int(input_box[3])),
                  (0, 255, 0), 2)
    cv2.putText(overlay_image, f"SAM Score: {score:.3f}", (int(input_box[0]), int(input_box[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 저장
    output_path = "debug_images/poc_mobilesam_result.jpg"
    cv2.imwrite(output_path, overlay_image)
    print(f"✅ 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()