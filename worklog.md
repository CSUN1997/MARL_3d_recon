## 11/27/2019 -- Sun
If you want to run codes, go to MARL_3D_RECON/MARL_grids, and run `python MARL.py`. The program first tries to connect to the video signal on the drone. A lot of error msgs will appear on the screen, don't worry just wait (Archana should know this :-D).

*Sometimes, the program went wrong and exit with the drone still hovering in the air. Run python test.py to make it land.*

There are some important things about the program and the drone:
- Some times the program doesn't exit nicely. You need to close the command line directly to terminate it. If the connection with the drone doesn't work, try reboot the drone.
- Everytime the drone finishes an episode, it will return to the starting point. However, because of the precision of the drone, it will not go to the original starting point precisely, it will always go up a little more. *Keep an eye on the dron and don't let it crush with the cameras in the drone lab*
- I run the algorithm in a fake environment (simulation). It works well. So don't worry about the AGENT. The only thing we need to do now is to figure out how to define a new reward. (This reward works really well btw.)
- Everytime when training terminates, the Q table will be saved to the Q_table.npy file. And everytime the program starts, it will try to read the Q_table.npy. If this file does not exist, it will start with an all-zero Q table.
- The state is defined as (last position, current position), which should be a 4 by 1 vector, because position has two axis (y, z). That's why Q table is of grid_size[0] * grid_size[1] * grid_size[0] * grid_size[1] * n_actions
- I defined a test function in MARL.py file. This function tests the agent's performance in the fake environment (simulation) and it works well. To test the agent in real settings, uncomment __line 290__.
- Right now I registered a image save function to save images, the default image save frequency is 30 frames per sec. This is too many. Thus, I leveraged __imagehash__ package in Python to make sure the incoming images are not too similar with the existing ones.

I didn't test the drone today (11/27/2019). There might be some small errors in the program but the main logic should work well.

*Two connect to two drones*
__The main idea is to assign the alienware's network adaptor to one virtual machine and assign the USB adaptor to another virtual machine.__
Open two virtual machines (which version doesn't matter as long as it can run pyparrot and python), I used ubuntu 14 and 16 last time. Plug in the USB network adaptor. Open wifi connection in the right bottom corner of Windows. There should be a dropdown box with Wifi and Wifi 2. Use Wifi to connect to one drone and Wifi 2 to connect to the other. I've already set the network settings of the vmware. You can run the programs now.