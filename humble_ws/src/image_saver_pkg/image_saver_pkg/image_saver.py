import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/rgb',  # Cambia se il tuo topic è diverso
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.image_saved = False

    def listener_callback(self, msg):
        if not self.image_saved:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                cv2.imwrite('immagine_salvata.png', cv_image)
                self.get_logger().info('✅ Immagine salvata come immagine_salvata.png')
                self.image_saved = True
            except Exception as e:
                self.get_logger().error(f'Errore nella conversione: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
