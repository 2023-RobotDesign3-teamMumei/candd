#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(message):
    rospy.loginfo("get message! [%s]", message.data) # ターミナルへの表示

rospy.init_node('listener')
sub = rospy.Subscriber('chatter', String, callback) # chatterというTopicを受信！受信したら上で定義したcallback関数を呼ぶ
rospy.spin()

