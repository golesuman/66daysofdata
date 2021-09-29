import math
phy=[15,12,8,8,7,7,7,6,5,3]
hist=[10,25,17,11,13,17,20,13,9,15]
phym=sum(phy)/len(phy)
hism=sum(hist)/len(hist)
diffphy=[phy_num-phym for phy_num in phy]
diffhist=[hist_num-hism for hist_num in hist]
mulxy=[]

for i in range(len(phy)):
    mulxy.append(diffphy[i]*diffhist[i])

diffphysqr=[(phy_num-phym)**2 for phy_num in phy]
diffhistsqr=[(hist_num-hism)**2 for hist_num in hist]

sum_cov=sum(mulxy)
variance_phy=math.sqrt(sum(diffphysqr))
variance_hist=math.sqrt(sum(diffhistsqr))
mul_variance=variance_phy*variance_hist

r=sum_cov/mul_variance
print(r)


