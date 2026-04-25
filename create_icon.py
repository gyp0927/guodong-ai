"""生成应用图标"""
from PIL import Image, ImageDraw, ImageFont
import os


def create_icon(output_path="icon.ico", size=256):
    """创建一个简洁的机器人图标"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 颜色（与你网站的主题色一致）
    bg_color = (16, 163, 127)    # #10a37f 绿色
    face_color = (255, 255, 255) # 白色

    # 外圆背景
    margin = size // 16
    bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(bbox, fill=bg_color)

    # 计算中心点
    cx, cy = size // 2, size // 2
    s = size  # 简写

    # 机器人脸 — 一个圆角矩形
    face_w = int(s * 0.50)
    face_h = int(s * 0.40)
    face_x1 = cx - face_w // 2
    face_y1 = int(s * 0.28)
    face_x2 = face_x1 + face_w
    face_y2 = face_y1 + face_h
    radius = face_w // 4
    draw.rounded_rectangle([face_x1, face_y1, face_x2, face_y2], radius=radius, fill=face_color)

    # 左眼
    eye_r = max(2, int(s * 0.045))
    left_eye_x = cx - int(s * 0.10)
    left_eye_y = cy - int(s * 0.02)
    draw.ellipse([left_eye_x - eye_r, left_eye_y - eye_r,
                  left_eye_x + eye_r, left_eye_y + eye_r], fill=bg_color)

    # 右眼
    right_eye_x = cx + int(s * 0.10)
    draw.ellipse([right_eye_x - eye_r, left_eye_y - eye_r,
                  right_eye_x + eye_r, left_eye_y + eye_r], fill=bg_color)

    # 嘴巴 — 微笑弧线
    mouth_y = cy + int(s * 0.08)
    mouth_w = int(s * 0.14)
    draw.arc([cx - mouth_w, mouth_y - mouth_w,
              cx + mouth_w, mouth_y + mouth_w],
             start=200, end=340, fill=bg_color, width=max(1, int(s * 0.025)))

    # 天线
    ant_x = cx
    ant_y1 = int(s * 0.18)
    ant_y2 = int(s * 0.28)
    ant_ball_r = max(2, int(s * 0.035))
    draw.line([(ant_x, ant_y1), (ant_x, ant_y2)], fill=face_color, width=max(1, int(s * 0.03)))
    draw.ellipse([ant_x - ant_ball_r, ant_y1 - ant_ball_r * 2,
                  ant_x + ant_ball_r, ant_y1], fill=face_color)

    return img


if __name__ == "__main__":
    # 生成多尺寸 ICO 文件
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    for sz in sizes:
        img = create_icon(size=sz)
        images.append(img)

    # 保存
    output = "icon.ico"
    images[0].save(output, format="ICO", sizes=[(s, s) for s in sizes], append_images=images[1:])
    print(f"图标已生成: {os.path.abspath(output)}")
