from point import Point


class Hand:

    def __init__(self, hand_landmarks, nose_point, image_height, image_width):
        self.hand_points = []
        for i in range(2, 21):
            new_hand_landmark_x = hand_landmarks.landmark[i].x - nose_point.x
            new_hand_landmark_y = hand_landmarks.landmark[i].y - nose_point.y
            new_hand_landmark_z = hand_landmarks.landmark[i].z
            self.hand_points.append(Point(new_hand_landmark_x, new_hand_landmark_y, new_hand_landmark_z))

    def print_hand(self):
        for i in self.hand_points:
            i.print_point()

    def get_timestep_data(self):
        hand = []
        for i in self.hand_points:
            hand.append(i.get_time_step_data())
        return ','.join(hand)






















        # self.thumb_mcp = Point(hand_landmarks.landmark[2].x * image_width, hand_landmarks.landmark[2].y * image_height)
        # self.thumb_ip = Point(hand_landmarks.landmark[3].x * image_width, hand_landmarks.landmark[3].y * image_height)
        # self.thumb_tip = Point(hand_landmarks.landmark[4].x * image_width, hand_landmarks.landmark[4].y * image_height)
        # self.index_mcp = Point(hand_landmarks.landmark[5].x * image_width, hand_landmarks.landmark[5].y * image_height)
        # self.index_pip = Point(hand_landmarks.landmark[6].x * image_width, hand_landmarks.landmark[6].y * image_height)
        # self.index_dip = Point(hand_landmarks.landmark[7].x * image_width, hand_landmarks.landmark[7].y * image_height)
        # self.index_tip = Point(hand_landmarks.landmark[8].x * image_width, hand_landmarks.landmark[8].y * image_height)
        # self.middle_mcp = Point(hand_landmarks.landmark[9].x * image_width, hand_landmarks.landmark[9].y * image_height)
        # self.middle_pip = Point(hand_landmarks.landmark[10].x * image_width, hand_landmarks.landmark[10].y * image_height)
        # self.middle_dip = Point(hand_landmarks.landmark[11].x * image_width, hand_landmarks.landmark[11].y * image_height)
        # self.middle_tip = Point(hand_landmarks.landmark[12].x * image_width, hand_landmarks.landmark[12].y * image_height)
        # self.ring_mcp = Point(hand_landmarks.landmark[13].x * image_width, hand_landmarks.landmark[13].y * image_height)
        # self.ring_pip = Point(hand_landmarks.landmark[14].x * image_width, hand_landmarks.landmark[14].y * image_height)
        # self.ring_dip = Point(hand_landmarks.landmark[15].x * image_width, hand_landmarks.landmark[15].y * image_height)
        # self.ring_tip = Point(hand_landmarks.landmark[16].x * image_width, hand_landmarks.landmark[16].y * image_height)
        # self.pinky_mcp = Point(hand_landmarks.landmark[17].x * image_width, hand_landmarks.landmark[17].y * image_height)
        # self.pinky_pip = Point(hand_landmarks.landmark[18].x * image_width, hand_landmarks.landmark[18].y * image_height)
        # self.pinky_dip = Point(hand_landmarks.landmark[19].x * image_width, hand_landmarks.landmark[19].y * image_height)
        # self.pinky_tip = Point(hand_landmarks.landmark[20].x * image_width, hand_landmarks.landmark[20].y * image_height)



    # def print_hand(self):
    #     print("Thumb: (", self.thumb_mcp.x, self.thumb_mcp.y, "), (", self.thumb_ip.x, self.thumb_ip.y, "), (", self.thumb_tip.x, self.thumb_tip.y, ")")
    #     print("Index: (", self.index_mcp.x, self.index_mcp.y, "), (", self.index_pip.x, self.index_pip.y, "), (", self.index_dip.x, self.index_dip.y, "), (", self.index_tip.x, self.index_tip.y, ")")
    #     print("Middle: (", self.middle_mcp.x, self.middle_mcp.y, "), (", self.middle_pip.x, self.middle_pip.y, "), (", self.middle_dip.x, self.middle_dip.y, "), (", self.middle_tip.x, self.middle_tip.y, ")")
    #     print("Ring: (", self.ring_mcp.x, self.ring_mcp.y, "), (", self.ring_pip.x, self.ring_pip.y, "), (", self.ring_dip.x, self.ring_dip.y, "), (", self.ring_tip.x, self.ring_tip.y, ")")
    #     print("Pinky: (", self.pinky_mcp.x, self.pinky_mcp.y, "), (", self.pinky_pip.x, self.pinky_pip.y, "), (", self.pinky_dip.x, self.pinky_dip.y, "), (", self.pinky_tip.x, self.pinky_tip.y, ")")
