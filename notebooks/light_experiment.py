# %%

# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
from timeit import default_timer as timer
from notebook_init import *
from tqdm import trange

out_root = Path('../results/')
makedirs(out_root, exist_ok=True)
B = 1

# Batch over latents if there are more latents than frames in strip
def createRandomSamples(inst, latent, z_comp, lat_stdev, lat_mean, sigma,
                            layer_start, layer_end, num_frames = 5, center = True):

    print ("Random Samples Debugging")
    #print (type(inst.model))

    num_frames = 5

    print ("Layers:")
    print (layer_start)
    print (layer_end)

    sigma_range = np.linspace(-sigma, sigma, num_frames, dtype=np.float32) #Grid values
    #sigma_range = sigma * np.random.randn(num_frames) #Sampling Gaussian Dist
    #sigma_range = np.random.uniform(-sigma, sigma, num_frames) #Sampling Uniform Dist

    #print (z_comp.shape)

    zs = latent

    normalize = lambda v: v / torch.sqrt(torch.sum(v ** 2, dim=-1, keepdim=True) + 1e-8)
    zeroing_offset_lat = 0

    frames = []

    if center: # Shift latent to lie on mean along given component
        dotp = torch.sum((zs - lat_mean) * normalize(z_comp), dim=-1, keepdim=True)
        zeroing_offset_lat = dotp * normalize(z_comp)

    for i in range(len(sigma_range)):
        s = sigma_range[i]

        with torch.no_grad():
            z = [zs] * inst.model.get_max_latents()  # one per layer

            #print ("Debugging Z")

            delta = z_comp * s * lat_stdev

            for k in range(layer_start, layer_end):
                z[k] = z[k] - zeroing_offset_lat + delta

            #print (len(z))
            #print (z[0].shape)

            img = inst.model.sample_np(z)

            frames.append(img)

    return frames

start = timer()

# %%

####################################### StyleGAN2 ffhq
# Model, layer, edit, layer_start, layer_end, class, sigma, idx, name, (example seeds)
configs = [
    #Lighting

    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', 13.0, 13, 'Bright BG vs FG', [798602383]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -8.0, 10, 'Sunlight in face', [798602383]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -15.0, 25, 'Light UD', [1382206226]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 18, 'ffhq', 5.0, 27, 'Overexposed', [1887645531]),
    ('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -14.0, 29, 'Highlights', [490151100, 1010645708]),

    #Experiments
    #('StyleGAN2', 'style', 'latent', 'w', 8, 8, 'ffhq', -8.0, 10, 'Sunlight in face', [798602383]), --> NO Change
    #('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -8.0, 10, 'Sunlight in face', [798602383]),
    #('StyleGAN2', 'style', 'latent', 'w', 7, 9, 'ffhq', -20.0, 58, 'Trimmed beard', [798602383]),

    #('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', -10.0, 25, 'Light UD', [798602383])
    #('StyleGAN2', 'style', 'latent', 'w', 8, 9, 'ffhq', 5.0, 13, 'Bright BG vs FG', [798602383]),
    #('StyleGAN2', 'style', 'latent', 'w', 8, 18, 'ffhq', 5.0, 27, 'Overexposed', [798602383])
]

has_gpu = torch.cuda.is_available()
device = torch.device('cuda' if has_gpu else 'cpu')

num_imgs_per_example = 1
num_seeds = 1

for config_id, (
model_name, layer, mode, latent_space, l_start, l_end, classname, sigma, idx, title, seeds) in enumerate(configs[:]):
    print(f'{model_name}, {layer}, {title}')

    inst = get_instrumented_model(model_name, classname, layer, device, inst=inst)  # reuse if possible
    model = inst.model

    model.truncation = 0.7
    model.use_w()

    # Load or compute decomposition
    config = Config(
        output_class=classname,
        model=model_name,
        layer=layer,
        estimator='ipca',
        use_w=True,
        n=1_000_000
    )

    dump_name = get_or_compute(config, inst)
    data = np.load(dump_name, allow_pickle=False)
    X_comp = data['act_comp']
    X_global_mean = data['act_mean']
    X_stdev = data['act_stdev']
    Z_global_mean = data['lat_mean']
    Z_comp = data['lat_comp']
    Z_stdev = data['lat_stdev']
    data.close()

    model.set_output_class(classname)
    feat_shape = X_comp[0].shape
    sample_dims = np.prod(feat_shape)

    # Transfer to GPU
    components = SimpleNamespace(
        X_comp=torch.from_numpy(X_comp).view(-1, *feat_shape).to('cuda').float(),  # -1, 1, C, H, W
        X_global_mean=torch.from_numpy(X_global_mean).view(*feat_shape).to('cuda').float(),  # 1, C, H, W
        X_stdev=torch.from_numpy(X_stdev).to('cuda').float(),
        Z_comp=torch.from_numpy(Z_comp).to('cuda').float(),
        Z_stdev=torch.from_numpy(Z_stdev).to('cuda').float(),
        Z_global_mean=torch.from_numpy(Z_global_mean).to('cuda').float(),
    )

    #Create random latent vectors
    max_seed = np.iinfo(np.int32).max
    seeds = np.concatenate((seeds, np.random.randint(0, max_seed, num_seeds)))
    seeds = seeds[:num_seeds].astype(np.int32)
    latents = [model.sample_latent(1, seed=s) for s in seeds]

    # Range is exclusive, in contrast to notation in paper
    edit_start = l_start
    edit_end = model.get_max_latents() if l_end == -1 else l_end

    # Preparing Output
    edit_name = prettify_name(title)
    output_dir = os.path.join(out_root, edit_name)

    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for id, latent in enumerate(latents):

        samples = createRandomSamples(inst, latent, components.Z_comp[idx], components.Z_stdev[idx],
                                       components.Z_global_mean, sigma, edit_start, edit_end)

        if id == 0:  # Show first person

            examples = np.hstack(pad_frames(samples[0:5]))

            plt.figure(figsize=(15, 15))
            plt.imshow(examples)
            plt.axis('off')
            plt.show()

        new_folder = os.path.join(output_dir, "person_{:04d}".format(id))
        os.makedirs(new_folder, exist_ok=True)

        for i, frame in enumerate(samples):
            file = "face_{:04d}.png".format(i)

            img = Image.fromarray(np.uint8(frame * 255))
            img.save(os.path.join(new_folder, file))

print('Done')
end = timer()
print("StyleGAN Data Generation: {:0.2f} (s)".format(end - start))