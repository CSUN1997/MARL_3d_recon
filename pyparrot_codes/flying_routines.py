def takeoff_land(bebopVision, args):
    #this function serves to take the drone off and then land it after hovering a bit.
    #simple procedure to check if drone is working correctly
    bebop = args[0]
    
    print("I am hovering!")
    bebop.smart_sleep(5)
    
    takeoff(bebop)
    land_and_disconnect(bebop, bebopVision)


def move_around(bebopVision, args):
    #get bebop
    bebop = args[0]

    #takeoff
    takeoff(bebop)
    
    #move right
    bebop.move_relative(0, -1, 0, 0) #moving left
    bebop.smart_sleep(1)
    bebop.move_relative(0, 0, -1, 0) #moving up
    bebop.smart_sleep(1)
    bebop.move_relative(0, 1, 0, 0) #moving right
    bebop.smart_sleep(1)
    bebop.move_relative(0, 0, 1, 0) #moving down
    bebop.smart_sleep(1)
    
    land_and_disconnect(bebop, bebopVision)
    
def reinforcement(grid, steps):
    for steps in yourfunction():
        move relative according to the function

def takeoff(bebop):
    #helper function for takeoff
    # takeoff
    bebop.safe_takeoff(5)
    
def land_and_disconnect(bebop, bebopVision):
    #helper function for landing
    if (bebopVision.vision_running):
        # land
        bebop.safe_land(5)

    print("Finishing demo and stopping vision")
    bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    print("disconnecting")
    bebop.disconnect()