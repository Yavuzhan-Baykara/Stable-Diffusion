import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

class SampleDiffusionProcessGenerator:
    def __init__(self, n_steps, t0, dt):
        self.n_steps = n_steps
        self.t0 = t0
        self.dt = dt

    def forward_diffusion_1d(self, x0, noise_strength_fn):
        x = np.zeros(self.n_steps + 1)
        x[0] = x0
        t = self.t0 + np.arange(self.n_steps + 1) * self.dt

        for i in range(self.n_steps):
            noise_strength = noise_strength_fn(t[i])
            x[i + 1] = x[i] + np.random.randn() * noise_strength

        return x, t

    def reverse_diffusion_1d(self, x0, score_fn, T, noise_strength_fn):
        x = np.zeros(self.n_steps + 1)
        x[0] = x0
        t = np.arange(self.n_steps + 1) * self.dt

        for i in range(self.n_steps):
            noise_strength = noise_strength_fn(T - t[i])
            score = score_fn(x[i], 0, noise_strength, T - t[i])
            x[i + 1] = x[i] + score * noise_strength**2 * self.dt + noise_strength * np.random.randn() * np.sqrt(self.dt)

        return x, t

    @staticmethod
    def noise_strength_constant(t):
        return 1

    @staticmethod
    def score_simple(x, x0, noise_strength, t):
        return - (x - x0) / ((noise_strength**2) * t)


class SamplerVisualization:
    def __init__(self, sample_diffusion_process_generator):
        self.sample_diffusion_process_generator = sample_diffusion_process_generator

    def visualize_diffusion(self, choice=0, num_tries=5):
        if choice == 0:
            diffusion_type = "Forward"
            diffusion_func = self.sample_diffusion_process_generator.forward_diffusion_1d
        elif choice == 1:
            diffusion_type = "Reverse"
            diffusion_func = self.sample_diffusion_process_generator.reverse_diffusion_1d
        else:
            raise ValueError("Invalid choice")

        plt.figure(figsize=(15, 5))
        for _ in range(num_tries):
            x0 = np.random.normal(loc=0, scale=T) if choice == 1 else 0
            if choice == 1:
                x, t = diffusion_func(x0=x0, score_fn=SampleDiffusionProcessGenerator.score_simple, T=T, noise_strength_fn=SampleDiffusionProcessGenerator.noise_strength_constant)
            else:
                x, t = diffusion_func(x0=x0, noise_strength_fn=SampleDiffusionProcessGenerator.noise_strength_constant)
            plt.plot(t, x)

        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Sample Value ($x$)", fontsize=20)
        plt.title(f"{diffusion_type} Diffusion Visualization", fontsize=20)
        plt.legend()
        plt.show()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="choice path")
    args = parser.parse_args()
    
    with open(args.file_path, "r") as f:
        data = json.load(f)
    
    n_steps = data["n_steps"]
    t0 = data["t0"]
    dt = data["dt"]
    T = data["T"]
    choice = data["choice"]

    sample_diffusion_process_generator = SampleDiffusionProcessGenerator(n_steps, t0, dt)
    sampler_visualizer = SamplerVisualization(sample_diffusion_process_generator)
    sampler_visualizer.visualize_diffusion(choice)
