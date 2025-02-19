from PIL import Image
import numpy as np
import torch

from settings import *


def get_image_output_name(labels, label_means, label_stds):
    if isinstance(labels, torch.Tensor):
        output_name = [x for x in labels[0].detach().cpu().numpy()]
    else:
        output_name = [x for x in labels]
    non_shift_labels = len(output_name) - len(SELECTED_INDICES)
    for x in range(non_shift_labels, len(output_name)):
        output_name[x] = output_name[x] * label_stds[x - non_shift_labels] + label_means[x - non_shift_labels]
    output_name[0] = output_name[0] / 2 + 0.5
    output_name[1] = output_name[1] / 2 + 0.5
    return "_".join(f"{output_name[x]:.2f}" if x < 2 else f"{int(round(output_name[x]))}" for x in range(len(output_name)))


def save_generator_image(generator_output, output_folder, epoch, labels, label_means, label_stds):
    output_name = get_image_output_name(labels, label_means, label_stds)
    fake_image_np = ((generator_output.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
    fake_image_pil = Image.fromarray(fake_image_np)
    fake_image_pil.save(f"{output_folder}/E{(epoch + 1):03}_{output_name}.png")  # Save fake image


def refresh_results(generator, epoch, num_epochs, last_epoch_time, loss_G, loss_D, labels, pred_real, fake_labels, pred_fake, delta_latent, delta_feature):
    current_time = datetime.now()
    time_since_last = (current_time - last_epoch_time).total_seconds()
    datetime_next = current_time + (current_time - last_epoch_time)

    film1_std_params = generator.film1.get_std_parameters().cpu().detach()
    film2_std_params = generator.film2.get_std_parameters().cpu().detach()
    film3_std_params = generator.film3.get_std_parameters().cpu().detach()

    print("\n" + "=" * 100)
    print(f"Time: {str(current_time)[:19]} | Epoch Time (s): {time_since_last:.2f} | Next Epoch: {str(datetime_next)[:19]}")
    print("-" * 100)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss_G: {loss_G.item():.3f}, Loss_D: {loss_D.item():.3f}")
    print("")
    print(f"G Realism Loss: {torch.abs(torch.ones_like(pred_fake[:, -1]) - pred_fake[:, -1]).mean().item():.3f} "
          f"Label Loss: [{', '.join([f'{torch.abs(fake_labels[:, x] - pred_fake[:, x]).mean().item():.3f}' for x in range(NUM_LABELS)])}]"
          f"Latent Variance: Delta Latent: {delta_latent} Delta Image: {delta_feature}")
    print(f"D Realism Loss: {torch.abs(torch.ones_like(pred_real[:, -1]) - pred_real[:, -1]).mean().item():.3f} "
          f"Label Loss: [{', '.join([f'{torch.abs(labels[:, x] - pred_real[:, x]).mean().item():.3f}' for x in range(NUM_LABELS)])}]")
    print("")
    print(
        f"G FiLM Gamma Weight Std: γ1 = {film1_std_params[0].item():.3f}, γ2 = {film2_std_params[0].item():.3f}, γ3 = {film3_std_params[0].item():.3f}")
    print(
        f"G FiLM Gamma Biases Std: γ1 = {film1_std_params[1].item():.3f}, γ2 = {film2_std_params[1].item():.3f}, γ3 = {film3_std_params[1].item():.3f}")
    print(
        f"G FiLM Beta Weights Std: β1 = {film1_std_params[2].item():.3f}, β2 = {film2_std_params[2].item():.3f}, β3 = {film3_std_params[2].item():.3f}")
    print(
        f"G FiLM Beta Biases Std: β1 = {film1_std_params[3].item():.3f}, β2 = {film2_std_params[3].item():.3f}, β3 = {film3_std_params[3].item():.3f}")
    print("")
    print("                                  ['Date', 'Time', 'Temp', 'Press', 'Dew', 'Hum', 'Dir', 'Alt']")
    print("Real Input Labels:               ", [f"{x:.1f}" for x in labels[0].cpu().numpy()])
    print("Predicted Labels (Real):         ", [f"{x:.1f}" for x in pred_real[0, :-1].detach().cpu().numpy()])
    print("Random Labels (Generator Input): ", [f"{x:.1f}" for x in fake_labels[0].cpu().numpy()])
    print("Predicted Labels (Fake):         ", [f"{x:.1f}" for x in pred_fake[0, :-1].detach().cpu().numpy()])
    print(f"Real Check (Real): {pred_real[0, -1].item():.1f} | (Fake): {pred_fake[0, -1].item():.1f}")
    print("-" * 100)
    print("")

    return last_epoch_time
