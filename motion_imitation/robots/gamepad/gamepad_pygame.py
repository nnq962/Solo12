import pygame

class PygameKeyboardControl:
    def __init__(self):
        # Khởi tạo pygame
        pygame.init()
        # Thiết lập một cửa sổ rất nhỏ để nhận lệnh bàn phím (cửa sổ này không thực sự hiển thị)
        self.screen = pygame.display.set_mode((300, 300))

        # Đặt các giá trị điều khiển
        self.vx = 0
        self.vy = 0
        self.wz = 0

    def get_command(self, time_since_reset):
        del time_since_reset
        # Lấy các sự kiện từ bàn phím
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Kiểm tra các phím đang được nhấn
        keys = pygame.key.get_pressed()

        # Di chuyển về phía trước (W) và lùi lại (S)
        if keys[pygame.K_w]:
            self.vx = 1
        elif keys[pygame.K_s]:
            self.vx = -1
        else:
            self.vx = 0

        # Di chuyển sang trái (A) và phải (D)
        if keys[pygame.K_a]:
            self.vy = 1
        elif keys[pygame.K_d]:
            self.vy = -1
        else:
            self.vy = 0

        # Xoay quanh trục Z (Q và E)
        if keys[pygame.K_q]:
            self.wz = 1
        elif keys[pygame.K_e]:
            self.wz = -1
        else:
            self.wz = 0

        # Emergency stop (phím Space)
        estop_flagged = keys[pygame.K_SPACE]

        return (self.vx, self.vy, 0), self.wz, estop_flagged
