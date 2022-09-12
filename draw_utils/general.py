import cv2


def draw_text(frame, text, x, y, color, thickness=1, font_scale=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_circle(frame, x, y, r, color, thickness=1):
    return cv2.circle(frame, (x, y), r, color, thickness)


def draw_arc(frame, x, y, r, start_angle, end_angle, color, thickness=1):
    return cv2.ellipse(frame, (x, y), (r, r), 0, start_angle, end_angle, color, thickness)


def draw_line(frame, x1, y1, x2, y2, color, thickness=1):
    return cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(frame, x1, y1, x2, y2, color, thickness=1):
    return cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)