#!/usr/bin/env python

import rospy

class HabitatHydraBridgeNode:
    """Node to handle publishing data from Habitat-Sim to Hydra-ROS."""

    def __init__(self, with_mesh=True):
        self._pub = rospy.Publisher('chatter', String, queue_size=10)

    def spin(self):
        """Wait until rospy is shutdown."""
        rospy.spin()


def main():
    """Start ROS and the node."""
    rospy.init_node("habitat_hydra_bridge_node")

    node = HabitatHydraBridgeNode()

    node.spin()


if __name__ == "__main__":
    main()
