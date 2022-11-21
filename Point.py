class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def print_point(self):
        print(str(self.x) + ", " + str(self.y) + ", " + str(self.z))

    def get_time_step_data(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.z)
