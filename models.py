import numpy as np
import matplotlib.pyplot as plt



class SampleDiffusionProcessGenerator:
    def __init__(self):
        pass

    def forward_diffusion_1d(self, n_steps:int, x0:int, t0:int, dt:float, noise_strength_fn):
        x = np.zeros(n_steps + 1)
        x[0] = x0
        t = t0 + np.arange(n_steps + 1) * dt

        for i in range(n_steps):
            noise_strength = noise_strength_fn(t[i])
            random_normal = np.random.randn()
            x[i + 1] = x[i] + random_normal * noise_strength

        return x, t

    def noise_strength_constant(self, t):
        return 1

    def reverse_diffusion_1d(self,):
        pass

    def score_simple(self, x, x0, noise_strength, t):
        score = - (x- x0) / ((noise_strength**2) * t)
        return score

def checksampler():
    sampleDiff = SampleDiffusionProcessGenerator()
    n_steps = 100
    t0 = 0
    dt= 0.1
    noise_strength_fn = sampleDiff.noise_strength_constant
    x0 = 0
    num_tries = 5
    
    plt.figure(figsize=(15, 5))
    for i in range(num_tries):
        x, t = sampleDiff.forward_diffusion_1d(n_steps=n_steps, x0=x0, t0=t0, dt=dt, noise_strength_fn=noise_strength_fn)
        plt.plot(t, x)
    
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Sample Value ($x$)", fontsize=20)
    plt.title("Forward Diffusion Visualization", fontsize=20)
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    checksampler()