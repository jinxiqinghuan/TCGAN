from tensorboard.backend.event_processing import event_accumulator

epoch = 300

#加载日志数据
ea1=event_accumulator.EventAccumulator(r'/root/cloud_hard_drive/project/pet2ct/runs/exp22/events.out.tfevents.1648216657.pod-ssh-1926') 
ea1.Reload()
print(ea1.scalars.Keys())

val_acc=ea1.scalars.Items('psnr')
print(len(val_acc))
print([(i.step,i.value) for i in val_acc])

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(6,4))
ax1=fig.add_subplot(111)
val_acc=ea1.scalars.Items('psnr')
val_acc = val_acc[:epoch]
ax1.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_psnr', color='tab:red')
ax1.set_xlim(0)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)

ax1.set_title("The val loss of vif.", fontsize=12)

# acc=ea.scalars.Items('val_mse')
# ax1.plot([i.step for i in acc],[i.value for i in acc],label='val_mse')
# ax1.set_xlabel("step")
# ax1.set_ylabel("")

#第二个数据
ea2=event_accumulator.EventAccumulator(r'/root/cloud_hard_drive/project/pet2ct/runs/exp24/events.out.tfevents.1648823812.pod-ssh-1926') 
ea2.Reload()
print(ea2.scalars.Keys())

ax2=fig.add_subplot(111)
val_acc=ea2.scalars.Items('psnr')
val_acc = val_acc[:epoch]
ax2.plot([i.step for i in val_acc],[i.value for i in val_acc],label='val_psnr', color='tab:blue')
ax2.set_xlim(0)
# ax2.set_xlabel("Epoch", fontsize=12)
# ax2.set_ylabel("Score", fontsize=12)

# ax2.set_title("The val loss of vif.", fontsize=12)

plt.legend(loc='lower right')
# plt.show()
plt.savefig("test.png")

