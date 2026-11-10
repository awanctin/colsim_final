class PID:
    def __init__(self, Kp, Ki, Kd):
        """Set the PID gains and reset the memory.

        Args:
            Kp (float): Proportional gain
            Ki (float): Integral gain
            Kd (float): Derivative gain
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        """Compute the PID command for one step.

        Args:
            error (float): Current error
            dt (float): Time step, s

        Returns:
            float: Command to apply
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
