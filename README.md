**objective:** To solve a finding an optimal path problem

There is an employee who lives in Azadi (point 1) in the [Path](https://www.google.com/maps/d/viewer?mid=1sYBzYAzJ99W9Le8IObYut6tvwy8QlJUJ&ll=35.755360553840816%2C51.393035300000065&z=12) and wants to go to the office in Tajrish (point 11) in the path. Based on the traffic and some restriction the numbers of routes that he can decide is limited and also has been shown in the link. The amount of fuel consumed and the time spent on a route depends on a number of factors, including the length of the path, the slope of the route, the traffic volume.

The employee wants to know which path is the lowst one in term of time and also the path with least fuel consumed. Furthermore the employee tries to minimize the function which is shown bellow:

C = Fuel + Time^2


With [SARSA](https://en.wikipedia.org/wiki/State–action–reward–state–action) and [Q-learning](https://en.wikipedia.org/wiki/Q-learning) algorithm an employee can find the best path depends on fuel consumption and time.
The [networkx](https://networkx.github.io) is used in this project.
Also here is a MapBuilder class to build the map which can use in this project.
[E-greedy](https://junedmunshi.wordpress.com/2012/03/30/how-to-implement-epsilon-greedy-strategy-policy/) in this work can not be used because it is hard to converge to a way at the end of the episode but [boltzmann](https://en.wikipedia.org/wiki/Boltzmann_distribution) policy with high temperature will converge to the best path.
