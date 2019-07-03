from object_remover import ObjectRemover
import numpy as np
import os
from PIL import Image


PATH = os.path.dirname(os.path.realpath(__file__))


def train(save_path=os.path.join(PATH, 'result')):
    fcn = ObjectRemover()
    mask_gt, training = fcn.set_placeholder()
    session = fcn.create_session()
    fcn.set_session(session=session)
    image_input, keep_probability, mask = fcn.generate_segmentation_mask()
    mask, mask_gt_new, mask_in_crop = fcn.mask_crop_bounding(mask=mask, mask_gt=mask_gt, image=image_input)
    predict, mask_optimised, mask_acc = fcn.mark_loss(mask=mask, mask_gt=mask_gt_new)

    fcn.run_session()
    for epoch in range(fcn.training_epochs):
        for batch in range(fcn.batch_num):
            offset = (batch * fcn.batch_size) % (fcn.training_data.shape[0] - fcn.batch_size)
            image_batch = fcn.training_data[offset: (offset+fcn.batch_size)]
            mask_batch = fcn.training_mask_gt[offset: (offset+fcn.batch_size)]
            _ = fcn.session.run([mask_optimised], feed_dict={
                                                            image_input: image_batch,
                                                            mask_gt: mask_batch,
                                                            training: True,
                                                            keep_probability: 0.5
                                                        })
            print('===================\nEpoch: {}'.format(str(epoch)))
            if not epoch%fcn.eval_step:
                train_loss, accuracy = [], []
                for batch in range(fcn.batch_num):
                    offset = (batch * fcn.batch_size) % (fcn.training_data.shape[0] - fcn.batch_size)
                    image_batch = fcn.training_data[offset: (offset + fcn.batch_size)]
                    mask_batch = fcn.training_mask_gt[offset: (offset + fcn.batch_size)]
                    pred, m, train_mask_l, train_mask_acc = fcn.session.run([mask_optimised], feed_dict={
                                                                    image_input: image_batch,
                                                                    mask_gt: mask_batch,
                                                                    training: True,
                                                                    keep_probability: 1
                                                                })
                    train_loss.append(train_mask_l)
                    accuracy.append(train_mask_acc)
                print("training loss: {}\ntraining accuracy: {}\n".format(np.mean(train_loss), np.mean(accuracy)))
                pred = fcn.onehot_output(prediction=pred)
                im = Image.fromarray(pred[0].astype('uint8'))
                im.save(os.path.join(save_path, '{}_test_pred.png'.format(str(epoch))))

                im = Image.fromarray(m[0].astype('uint8'))
                im.save(os.path.join(save_path, '{}_test_ori.png'.format(str(epoch))))
                fcn.save(os.path.join(save_path, 'model/fcn_{}'.format(str(epoch))))

    fcn.save(os.path.join(save_path, 'model/fcn_final'))


if __name__ == '__init__':
    train()
