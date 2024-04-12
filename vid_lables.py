import cv2

text_y = 25


def dusty_labels(frame, mean, blurry):
    text = "Dusty: {:.4f}" if blurry else "Not Dusty: {:.4f}"
    text = text.format(mean)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)


def dust(frame, mean, dusty):
    dusty_text = f"Dusty: {mean:.4f}" if dusty else f"Not Dusty ({mean:.4f})"
    dusty_color = (0, 0, 255) if dusty else (0, 255, 0)

    cv2.putText(frame, dusty_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dusty_color, 1)


def timestamp(frame, ts):
    cv2.putText(frame, f"TS(s): {ts / 1000:.3f}", (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


def display(frame, bridge_text, roi_key, roi_points):
    bridge_color = (0, 0, 255) if "Bridge" in bridge_text else (0, 255, 0)
    cv2.putText(frame, bridge_text, (10, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bridge_color, 1)

    cv2.putText(frame, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def centre_labels(cX, cY, frame_mask, roi_key):

    cv2.circle(frame_mask, (cX, cY), 2, (255, 255, 255), 1)
    cv2.putText(frame_mask, roi_key, (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_roi_poly(frame, roi_key, roi_points):
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 1)
    cv2.putText(frame, roi_key, (roi_points[1:2, 0:1][-1][-1], roi_points[1:2, 1:][-1][-1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # x1, y1 = roi_points[0]
    # x2, y2 = roi_points[1]
    # x3, y3 = roi_points[2]
    # x4, y4 = roi_points[3]
    #
    # y2 = int(((y1 - y2) / 2) + y2)
    # bl_quadrant_pts = np.array([[x1, y1], [x1, y2], [(x1 + x4) // 2, y2], [(x1 + x4) // 2, y4]])
    # tl_quadrant_pts = np.array([[x1, int(((y1 - y2) / 2) + y2)], [x2, y2], [(x2 + x3) // 2, y2], [(x1 + x4) // 2, y2]])
    # tr_quadrant_pts = np.array([[(x2 + x3) // 2, y2], [(x2 + x3) // 2, y2], [x3, y3], [x3, int(((y1 - y2) / 2) + y3)]])
    # br_quadrant_pts = np.array([[(x1 + x4) // 2, y4], [(x1 + x4) // 2, y2], [x3, y2], [x4, y4]])
    # #print("bl", bl_quadrant_pts, "tl", tl_quadrant_pts, "tr", tr_quadrant_pts, "br",br_quadrant_pts)
    # return tl_quadrant_pts, tr_quadrant_pts, bl_quadrant_pts, br_quadrant_pts
