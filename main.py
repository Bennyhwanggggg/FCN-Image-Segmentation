from object_remover import ObjectRemover


def train():
    fcn = ObjectRemover()
    mask_gt, training = fcn.set_placeholder()
    session = fcn.create_session()
    fcn.set_session(session=session)
    image_input, keep_probability, mask = fcn.generate_segmentation_mask()
    mask, mask_gt_new, mask_in_crop = fcn.mask_crop_bounding(mask=mask, mask_gt=mask_gt, image=image_input)
    predict, mask_optimised = fcn.mark_loss(mask=mask, mask_gt=mask_gt_new)




if __name__ == '__init__':
    train()