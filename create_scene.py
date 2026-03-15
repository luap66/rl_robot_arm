"""
Create a MuJoCo XML scene with Conveyor Belt and Panda Robot
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def create_panda_with_conveyor_xml():
    """
    Create MuJoCo XML that combines the Panda robot with a conveyor belt
    """
    
    xml_content = """
<mujoco model="panda_with_conveyor">
    <compiler angle="radian" meshdir="assets"/>
    
    <include file="/home/paulw/projects/mujoco_menagerie/franka_emika_panda/panda.xml"/>
</mujoco>
    """
    
    return xml_content


def create_conveyor_scene():
    """
    Create a standalone scene with conveyor belt
    """
    
    xml_content = """<?xml version="1.0" ?>
<mujoco model="conveyor_belt_scene">
    <compiler angle="radian" meshdir="assets" autolimits="true"/>
    
    <option gravity="0 0 -9.81"/>
    
    <asset>
        <mesh name="conveyor_belt" file="conveyor_belt.stl" scale="0.001 0.001 0.001"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    </asset>
    
    <worldbody>
        <!-- Ground -->
        <geom name="ground" pos="0 0 -1" size="5 5 0.5" type="plane" material="grid"/>
        
        <!-- Lighting -->
        <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        
        <!-- Conveyor Belt (STL mesh) -->
        <body name="conveyor_base" pos="0 0 0">
            <inertial pos="0 0 0" mass="50"/>
            <geom name="conveyor_visual" type="mesh" mesh="conveyor_belt" 
                   pos="0 0 0" rgba="0.5 0.3 0.2 1"/>
            <geom name="conveyor_collision" type="box" 
                   pos="0 0 0.4" size="1.1 0.11 0.05" rgba="0.5 0.3 0.2 0.3"/>
        </body>
        
        <!-- Panda Robot -->
        <body name="panda_link0" pos="0 -0.5 0">
            <!-- Link 0 (base) -->
            <inertial mass="2.5" pos="0 0 0" 
              fullinertia="0.005 0.005 0.005 0 0 0"/>
            <geom name="panda_link0" pos="0 0 0" size="0.075" type="sphere" rgba="0.8 0.8 0.8 1"/>
            
            <!-- Joint 1 -->
            <joint name="joint1" pos="0 0 0.333" axis="0 0 1" type="hinge" 
                   range="-2.8973 2.8973" damping="0.3"/>
            <body name="panda_link1" pos="0 0 0.333">
                <inertial mass="2.7" pos="0 -0.0324 0" 
                  fullinertia="0.01 0.01 0.005 0 0 0"/>
                <geom name="panda_link1" pos="0 -0.0324 0" size="0.06 0.03" type="capsule" rgba="0.8 0.2 0.2 1"/>
                
                <!-- Joint 2 -->
                <joint name="joint2" pos="0 -0.0324 0" axis="0 1 0" type="hinge" 
                       range="-1.7628 1.7628" damping="0.3"/>
                <body name="panda_link2" pos="0 -0.0324 0">
                    <inertial mass="1.2" pos="0 0.0496 0" 
                      fullinertia="0.008 0.008 0.004 0 0 0"/>
                    <geom name="panda_link2" pos="0 0.0496 0" size="0.05 0.025" type="capsule" rgba="0.8 0.2 0.2 1"/>
                    
                    <!-- Joint 3 -->
                    <joint name="joint3" pos="0 0.0496 0" axis="0 0 1" type="hinge" 
                           range="-2.8973 2.8973" damping="0.3"/>
                    <body name="panda_link3" pos="0 0.0496 0">
                        <inertial mass="1.3" pos="0 0.0331 0" 
                          fullinertia="0.008 0.008 0.004 0 0 0"/>
                        <geom name="panda_link3" pos="0 0.0331 0" size="0.05 0.025" type="capsule" rgba="0.2 0.8 0.2 1"/>
                        
                        <!-- Joint 4 -->
                        <joint name="joint4" pos="0 0.0331 0" axis="0 1 0" type="hinge" 
                               range="-3.0718 -0.0698" damping="0.3"/>
                        <body name="panda_link4" pos="0 0.0331 0">
                            <inertial mass="1.425" pos="0 -0.0054 0" 
                              fullinertia="0.01 0.01 0.005 0 0 0"/>
                            <geom name="panda_link4" pos="0 -0.0054 0" size="0.04 0.02" type="capsule" rgba="0.2 0.8 0.2 1"/>
                            
                            <!-- Joint 5 -->
                            <joint name="joint5" pos="0 -0.0054 0" axis="0 0 1" type="hinge" 
                                   range="-2.8973 2.8973" damping="0.3"/>
                            <body name="panda_link5" pos="0 -0.0054 0">
                                <inertial mass="1.1" pos="0 0.075 0" 
                                  fullinertia="0.005 0.005 0.003 0 0 0"/>
                                <geom name="panda_link5" pos="0 0.075 0" size="0.04 0.02" type="capsule" rgba="0.2 0.2 0.8 1"/>
                                
                                <!-- Joint 6 -->
                                <joint name="joint6" pos="0 0.075 0" axis="0 1 0" type="hinge" 
                                       range="-0.0175 3.7525" damping="0.3"/>
                                <body name="panda_link6" pos="0 0.075 0">
                                    <inertial mass="0.735" pos="0 0 0" 
                                      fullinertia="0.003 0.003 0.002 0 0 0"/>
                                    <geom name="panda_link6" pos="0 0 0" size="0.03 0.015" type="capsule" rgba="0.2 0.2 0.8 1"/>
                                    
                                    <!-- Joint 7 -->
                                    <joint name="joint7" pos="0 0 0" axis="0 0 1" type="hinge" 
                                           range="-2.8973 2.8973" damping="0.3"/>
                                    <body name="panda_link7" pos="0 0 0">
                                        <inertial mass="0.2" pos="0 0 0.1" 
                                          fullinertia="0.0005 0.0005 0.0005 0 0 0"/>
                                        <geom name="panda_link7" pos="0 0 0.05" size="0.025" type="sphere" rgba="0.8 0.8 0.2 1"/>
                                        
                                        <!-- End effector -->
                                        <site name="end_effector" pos="0 0 0.1" size="0.01" type="sphere"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Panda arm actuators -->
        <velocity name="actuator1" joint="joint1" ctrlrange="-2.175 2.175"/>
        <velocity name="actuator2" joint="joint2" ctrlrange="-2.175 2.175"/>
        <velocity name="actuator3" joint="joint3" ctrlrange="-2.175 2.175"/>
        <velocity name="actuator4" joint="joint4" ctrlrange="-2.175 2.175"/>
        <velocity name="actuator5" joint="joint5" ctrlrange="-2.61 2.61"/>
        <velocity name="actuator6" joint="joint6" ctrlrange="-2.61 2.61"/>
        <velocity name="actuator7" joint="joint7" ctrlrange="-2.61 2.61"/>
    </actuator>
</mujoco>
    """
    
    return xml_content


if __name__ == "__main__":
    # Create conveyor scene file
    scene_xml = create_conveyor_scene()
    
    with open("conveyor_scene.xml", "w") as f:
        f.write(scene_xml)
    
    print("✓ Created conveyor_scene.xml")
