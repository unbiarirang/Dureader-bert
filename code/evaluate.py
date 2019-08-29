import torch

def evaluate(model, dev_data, device, gradient_accumulation_steps):
    total, losses = 0.0, []

    with torch.no_grad():
        model.eval()
        for batch in dev_data:

            input_ids, input_mask,segment_ids, start_positions, end_positions = batch.input_ids, batch.input_mask, batch.segment_ids, batch.start_position, batch.end_position
            loss, _, _ = model(input_ids.to(device), \
                                     segment_ids.to(device), input_mask.to(device), start_positions.to(device), end_positions.to(device))
            loss = loss / gradient_accumulation_steps
            losses.append(loss.sum().item())

        for i in losses:
            total += i
        with open("./log", 'a') as f:
            f.write("eval_loss: " + str(total / len(losses)) + "\n")

        return total / len(losses)
