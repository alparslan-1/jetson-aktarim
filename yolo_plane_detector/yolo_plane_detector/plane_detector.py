#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class PlaneDetectorNode(Node):
    def __init__(self):
        super().__init__('plane_detector_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
            
        self.publisher_ = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.bridge = CvBridge()
        
        # Modelinin dosya yolu
        model_path = '/home/emrullah/ros2_ws/src/yolo_plane_detector/models/best.pt'
        self.model = YOLO(model_path) 
        
        self.get_logger().info("YOLOv11 Uçak Tanıma Düğümü Başlatıldı!")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüştürme hatası: {e}")
            return

        results = self.model(cv_image, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.publisher_.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Mesaj yayınlama hatası: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PlaneDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()