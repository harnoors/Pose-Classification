import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import Float32MultiArray

# Subscriber for pose_classifiaction channel
# you can add code to do something with the predicted results
# Eg: Issue varius commnads to a drone or rover

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(Float32MultiArray,'pose_classification',self.cl_callback, 10)
        self.subscription

    async def cl_callback(self, msg):
        msg = msg.data
        result = int(msg[0])
        resultPr = float(msg[1])
        ## Use result and probability to make decesions 
        print("the predicted class is: ", int(result),"  with probability:", resultPr)
         


rclpy.init(args = None)
minimal_subscriber = MinimalSubscriber()   

def main():
    rclpy.spin(minimal_subscriber)

if __name__ == "__main__":
    main()
