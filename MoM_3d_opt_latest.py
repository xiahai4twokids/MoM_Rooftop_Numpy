
# coding: utf-8
import numpy as np
import scipy
import utilities_3d as utilities
import time
# In[114]:
# 循环迭代 产生矩阵
# 下面的迭代适用3Ｄ问题


def MatrixFilling_opt(mesh,rooftops):
    try:
        # 矩阵化的x0_m　[坐标，区间段]
        x0_m = np.array([meshcell[0] for meshcell in mesh]).transpose()
        # 矩阵化的ｌｍ　[区间段]
        l_m = np.array([meshcell[2] for meshcell in mesh])
        # 矩阵化的ａ_m　[区间段]
        a_m = np.array([meshcell[3] for meshcell in mesh])        
        # 矩阵化的hl_m　[坐标，区间段]
        hl_m = np.array([meshcell[1]-meshcell[0] for meshcell in mesh])
        hl_m = hl_m.transpose()
        hl_m = np.divide(hl_m,l_m)
        # 矩阵化的ha_m　[坐标，区间段]
        ha_m = np.array([meshcell[4] for meshcell in mesh])
        ha_m = ha_m.transpose()
        # 标准高斯点
        W_x,P_x = utilities.getWeightAndPoint_S(2)
        W_y,P_y = utilities.getWeightAndPoint_S(1)
        # 存储每个矩形上的高斯点 [高斯点个数，坐标，区间段]
        PP_x, PP_y = np.meshgrid(P_x,P_y)
        WW_x, WW_y = np.meshgrid(W_x,W_y)
        PP_x = PP_x.reshape([1,-1])
        PP_y = PP_y.reshape([1,-1])
        WW_x = WW_x.reshape([1,-1])
        WW_y = WW_y.reshape([1,-1])
        
        shift_x = np.array([np.multiply(hl_m[:,ii:ii+1],np.multiply(l_m[ii:ii+1],(PP_x+1.)*0.5)) for ii in xrange(hl_m.shape[1])])
        shift_y = np.array([np.multiply(ha_m[:,ii:ii+1],np.multiply(a_m[ii:ii+1],PP_y*0.5)) for ii in xrange(hl_m.shape[1])])        
        Points = np.add(x0_m,  (shift_x + shift_y).transpose([2,1,0]))
        # 存放＋矩形的编号
        Seg_Pos = np.array([rooftopcell['pos'][0] for rooftopcell in rooftops])
        # 存放——矩形的编号
        Seg_Neg = np.array([rooftopcell['neg'][0] for rooftopcell in rooftops])
                 
    except ValueError as ve:
        print ve
        print hl_m.shape
        print l_m.shape
        print PP_x.shape
        raise
    except Exception as e:
        print e
        raise
            
    try:
        Z = scipy.zeros((len(rooftops),len(rooftops)),dtype=scipy.complex64) #阻抗矩阵
    except Exception as e:
        print e
        raise
        
    for n,edge_n in enumerate(rooftops): # 遍历所有测试
        # 计算pos线段的测试
        try:
            seg_n_id_pos = edge_n['pos'][0]
            seg_n_pos = mesh[seg_n_id_pos]
            an_pos = seg_n_pos[3] #宽度
            ln_pos = seg_n_pos[2] # 线段的长度
            hln_pos = (seg_n_pos[1]-seg_n_pos[0])/ln_pos # 计算测试的向量
            r_center_pos = (seg_n_pos[1]+seg_n_pos[0])*0.5
        except Exception as e:
            print e
            raise   
        try:
            seg_n_id_neg = edge_n['neg'][0]
            seg_n_neg = mesh[seg_n_id_neg]
            an_neg = seg_n_neg[3] #宽度
            ln_neg = seg_n_neg[2] # 线段的长度
            hln_neg = (seg_n_neg[1]-seg_n_neg[0])/ln_neg # 计算测试的向量
            r_center_neg = (seg_n_neg[1]+seg_n_neg[0])*0.5
        except Exception as e:
            print e
            raise

        try:
            # 计算积分点到外部点的矢径　[高斯点个数，坐标，区间段]
            RR_pos = np.subtract(Points, r_center_pos.reshape([3,-1]))
            RR_neg = np.subtract(Points, r_center_neg.reshape([3,-1]))
            # 计算积分点到外部点的距离　[高斯点个数，区间段]
            DDist_pos = np.sqrt(np.sum(np.multiply(RR_pos,RR_pos),axis=1))
            DDist_neg = np.sqrt(np.sum(np.multiply(RR_neg,RR_neg),axis=1))
            # 计算格林函数　[高斯点个数，区间段]
            ejkR_R_pos = np.divide(np.exp(-1j*wavenumber*DDist_pos),DDist_pos)
            ejkR_R_neg = np.divide(np.exp(-1j*wavenumber*DDist_neg),DDist_neg)
            # 计算格林函数的数值积分 [区间段]
            ejkR_RXw_pos = np.multiply(np.multiply(WW_x.transpose(),WW_y.transpose()),ejkR_R_pos)
            ejkR_RXw_neg = np.multiply(np.multiply(WW_x.transpose(),WW_y.transpose()),ejkR_R_neg)
            Int_ejkR_R_pos = np.multiply(np.multiply(np.sum(ejkR_RXw_pos,axis=0), l_m*0.5), a_m*0.5)# 0.5l_m 和0.5a_m是高斯积分权重的缩放因子
            Int_ejkR_R_neg = np.multiply(np.multiply(np.sum(ejkR_RXw_neg,axis=0), l_m*0.5), a_m*0.5)
            # 计算x_ejkR_RXw　用于＋矩形
            x_ejkR_RXw_pos = np.multiply((PP_x+1.).transpose(),ejkR_RXw_pos)
            x_ejkR_RXw_neg = np.multiply((PP_x+1.).transpose(),ejkR_RXw_neg)
            Int_x_ejkR_RXw_pos = np.multiply(np.multiply(np.sum(x_ejkR_RXw_pos,axis=0), 0.25*l_m**2), a_m*0.5)# 0.25l_m是权重缩放因子和位置缩放因子的乘积
            Int_x_ejkR_RXw_neg = np.multiply(np.multiply(np.sum(x_ejkR_RXw_neg,axis=0), 0.25*l_m**2), a_m*0.5)
            # 计算l-x_ejkR_RXw 用于﹣矩形
            Int_l_x_ejkR_RXw_pos = np.multiply(Int_ejkR_R_pos, l_m) - Int_x_ejkR_RXw_pos
            Int_l_x_ejkR_RXw_neg = np.multiply(Int_ejkR_R_neg, l_m) - Int_x_ejkR_RXw_neg
            
            # 计算电标位
            Phi_Inter_pos = np.divide(Int_ejkR_R_pos,l_m)
            Phi_Inter_neg = np.divide(Int_ejkR_R_neg,l_m)
            Phi_Inert_pos_Full = Phi_Inter_pos[Seg_Pos]-Phi_Inter_pos[Seg_Neg]
            Phi_Inert_neg_Full = Phi_Inter_neg[Seg_Pos]-Phi_Inter_neg[Seg_Neg]
            Phi_Inert_pos_Full = np.divide(Phi_Inert_pos_Full, ln_pos) # 乘以外面测试函数
            Phi_Inert_neg_Full = np.divide(Phi_Inert_neg_Full, ln_neg) # 乘以外面测试函数
            Phi_Inert_pos_Full = np.divide(Phi_Inert_pos_Full, wavenumber**2) # 除以k**
            Phi_Inert_neg_Full = np.divide(Phi_Inert_neg_Full, wavenumber**2) # 除以k**
            # 计算磁矢量位
            hl_n_Dot_hlm_pos = np.sum(np.multiply(hln_pos.reshape([3,1]),hl_m),axis=0)
            hl_n_Dot_hlm_neg = np.sum(np.multiply(hln_neg.reshape([3,1]),hl_m),axis=0)
            
            A_Inter_Pos = np.divide(Int_x_ejkR_RXw_pos,l_m)
            A_Inter_Pos = np.multiply(A_Inter_Pos,hl_n_Dot_hlm_pos)
            A_Inter_Neg = np.divide(Int_l_x_ejkR_RXw_pos,l_m)
            A_Inter_Neg = np.multiply(A_Inter_Neg,hl_n_Dot_hlm_pos)
            A_Inter_pos_Full = A_Inter_Pos[Seg_Pos]+A_Inter_Neg[Seg_Neg]
            A_Inter_pos_Full = np.multiply(A_Inter_pos_Full,0.5)# 乘以外部的测试函数
            
            A_Inter_Pos = np.divide(Int_x_ejkR_RXw_neg,l_m)
            A_Inter_Pos = np.multiply(A_Inter_Pos,hl_n_Dot_hlm_neg)
            A_Inter_Neg = np.divide(Int_l_x_ejkR_RXw_neg,l_m)
            A_Inter_Neg = np.multiply(A_Inter_Neg,hl_n_Dot_hlm_neg)
            A_Inter_neg_Full = A_Inter_Pos[Seg_Pos]+A_Inter_Neg[Seg_Neg]
            A_Inter_neg_Full = np.multiply(A_Inter_neg_Full,0.5)# 乘以外部的测试函数
            
            Z[n,:]= np.multiply(np.subtract(A_Inter_pos_Full, Phi_Inert_pos_Full), ln_pos*an_pos)
            Z[n,:] = Z[n,:] + np.multiply(np.add(A_Inter_neg_Full, Phi_Inert_neg_Full), ln_neg*an_neg)
            
            # 计算负的测试矩形
            
        except ValueError as ve:
            print ve
            raise
        except Exception as e:
            print e
            raise
    return Z         

def MatrixFilling_opt_new(mesh,rooftops):
    try:
        # 矩阵化的x0_m　[坐标，区间段]
        x0_m = np.array([meshcell[0] for meshcell in mesh]).transpose()
        # 矩阵化的ｌｍ　[区间段]
        l_m = np.array([meshcell[2] for meshcell in mesh])
        # 矩阵化的ａ_m　[区间段]
        a_m = np.array([meshcell[3] for meshcell in mesh])     
        # 存储每个矩形的中心点
        Centers = np.array([meshcell[0]+meshcell[1] for meshcell in mesh]).transpose()*0.5
        # 矩阵化的hl_m　[坐标，区间段]
        hl_m = np.array([meshcell[1]-meshcell[0] for meshcell in mesh])
        hl_m = hl_m.transpose()
        hl_m = np.divide(hl_m,l_m)
        # 矩阵化的ha_m　[坐标，区间段]
        ha_m = np.array([meshcell[4] for meshcell in mesh])
        ha_m = ha_m.transpose()
        # 标准高斯点
        W_x,P_x = utilities.getWeightAndPoint_S(2)
        W_y,P_y = utilities.getWeightAndPoint_S(1)
        # 存储每个矩形上的高斯点 [高斯点个数，坐标，区间段]
        PP_x, PP_y = np.meshgrid(P_x,P_y)
        WW_x, WW_y = np.meshgrid(W_x,W_y)
        PP_x = PP_x.reshape([1,-1])
        PP_y = PP_y.reshape([1,-1])
        WW_x = WW_x.reshape([1,-1])
        WW_y = WW_y.reshape([1,-1])
        shift_x = np.array([np.multiply(hl_m[:,ii:ii+1],np.multiply(l_m[ii:ii+1],(PP_x+1.)*0.5)) for ii in xrange(hl_m.shape[1])])
        shift_y = np.array([np.multiply(ha_m[:,ii:ii+1],np.multiply(a_m[ii:ii+1],PP_y*0.5)) for ii in xrange(hl_m.shape[1])])        
        Points = np.add(x0_m,  (shift_x + shift_y).transpose([2,1,0]))
        # 存放＋矩形的编号
        Seg_Pos = np.array([rooftopcell['pos'][0] for rooftopcell in rooftops])
        # 存放——矩形的编号
        Seg_Neg = np.array([rooftopcell['neg'][0] for rooftopcell in rooftops])
                 
    except ValueError as ve:
        print ve
        raise
    except Exception as e:
        print e
        raise
            
    try:
        Z = scipy.zeros((len(rooftops),len(rooftops)),dtype=scipy.complex64) #阻抗矩阵
    except Exception as e:
        print e
        raise
        
    try:
        half_lm = l_m*0.5
        half_am = a_m*0.5
        jK = 1j*wavenumber
        jK_lm_am = -jK*l_m*a_m
        jK_lm2_am_half = jK_lm_am*half_lm
        half_lm_am = np.multiply(half_lm,half_am)
        half_lm__2 = half_lm**2
        half_lm__2_am = np.multiply(half_lm__2,half_am)
    except Exception as e:
        print e
        raise
        
    for n,edge_n in enumerate(rooftops): # 遍历所有测试
        # 计算pos线段的测试
        try:
            seg_n_id_pos = edge_n['pos'][0]
            seg_n_pos = mesh[seg_n_id_pos]
            an_pos = seg_n_pos[3] #宽度
            ln_pos = seg_n_pos[2] # 线段的长度
            hln_pos = (seg_n_pos[1]-seg_n_pos[0])/ln_pos # 计算测试的向量
            r_center_pos = (seg_n_pos[1]+seg_n_pos[0])*0.5
        except Exception as e:
            print e
            raise   
        try:
            seg_n_id_neg = edge_n['neg'][0]
            seg_n_neg = mesh[seg_n_id_neg]
            an_neg = seg_n_neg[3] #宽度
            ln_neg = seg_n_neg[2] # 线段的长度
            hln_neg = (seg_n_neg[1]-seg_n_neg[0])/ln_neg # 计算测试的向量
            r_center_neg = (seg_n_neg[1]+seg_n_neg[0])*0.5
        except Exception as e:
            print e
            raise
        try:
            r_center = np.array([r_center_pos,r_center_neg])
            pass
        except Exception as e:
            print e
            print r_center
            raise

        try:
            # 计算积分点到外部点的矢径　[高斯点个数，坐标，区间段]
            RR_pos = np.subtract(Points, r_center_pos.reshape([3,-1]))
            RR_neg = np.subtract(Points, r_center_neg.reshape([3,-1]))
            
            RR0_pos = np.subtract(Centers, r_center_pos.reshape([3,-1]))
            RR0_neg = np.subtract(Centers, r_center_neg.reshape([3,-1]))

            # 计算积分点到外部点的距离　[高斯点个数，区间段]
            DDist_pos = np.sqrt(np.sum(np.multiply(RR_pos,RR_pos),axis=1))
            DDist_neg = np.sqrt(np.sum(np.multiply(RR_neg,RR_neg),axis=1))
            
            DDist0_pos = np.sqrt(np.sum(np.multiply(RR0_pos,RR0_pos),axis=0)).reshape([1,-1])
            DDist0_neg = np.sqrt(np.sum(np.multiply(RR0_neg,RR0_neg),axis=0)).reshape([1,-1])
            # 计算格林函数　[高斯点个数，区间段]
#            ejkR_R_pos = np.divide(np.exp(-1j*wavenumber*DDist_pos),DDist_pos)
#            ejkR_R_neg = np.divide(np.exp(-1j*wavenumber*DDist_neg),DDist_neg)
                     
            # e^{-jkR_0}
            ejkR0_pos = np.exp(-jK*DDist0_pos)            
            ejkR0_neg = np.exp(-jK*DDist0_neg)

#            raise
            # 计算格林函数的数值积分 [区间段]
            # Int_\dfrac{1}{R} ds \time (1+jkR_0) -jKArea
            _1_RXw_pos = np.divide(np.multiply(WW_x.transpose(),WW_y.transpose()),DDist_pos)
            _1_RXw_neg = np.divide(np.multiply(WW_x.transpose(),WW_y.transpose()),DDist_neg)
            Int_1_R_pos = np.multiply(np.sum(_1_RXw_pos,axis=0), half_lm_am)# 0.5l_m 和0.5a_m是高斯积分权重的缩放因子
            Int_1_R_neg = np.multiply(np.sum(_1_RXw_neg,axis=0), half_lm_am)
            
            temp1_pos = 1.+jK*DDist0_pos
            temp1_neg = 1.+jK*DDist0_neg

            Int_1_R_pos = np.add( np.multiply(Int_1_R_pos, temp1_pos), jK_lm_am)
            Int_1_R_neg = np.add( np.multiply(Int_1_R_neg, temp1_neg), jK_lm_am)
            Int_ejkR_R_pos = np.multiply(Int_1_R_pos, ejkR0_pos)
            Int_ejkR_R_neg = np.multiply(Int_1_R_neg, ejkR0_neg)
            # 计算x_ejkR_RXw　用于＋矩形
            x_RXw_pos = np.multiply((PP_x+1.).transpose(),_1_RXw_pos)
            x_RXw_neg = np.multiply((PP_x+1.).transpose(),_1_RXw_neg)

            Int_x_RXw_pos = np.multiply(np.sum(x_RXw_pos,axis=0), half_lm__2_am)# 0.25l_m是权重缩放因子和位置缩放因子的乘积
            Int_x_RXw_neg = np.multiply(np.sum(x_RXw_neg,axis=0), half_lm__2_am)
            
            Int_x_RXw_pos = np.add( np.multiply(Int_x_RXw_pos, temp1_pos), jK_lm2_am_half )
            Int_x_RXw_neg = np.add( np.multiply(Int_x_RXw_neg, temp1_neg), jK_lm2_am_half )
            Int_x_ejkR_RXw_pos = np.multiply(Int_x_RXw_pos, ejkR0_pos)
            Int_x_ejkR_RXw_neg = np.multiply(Int_x_RXw_neg, ejkR0_neg)
            # 计算l-x_ejkR_RXw 用于﹣矩形
            Int_l_x_ejkR_RXw_pos = np.multiply(Int_ejkR_R_pos, l_m) - Int_x_ejkR_RXw_pos
            Int_l_x_ejkR_RXw_neg = np.multiply(Int_ejkR_R_neg, l_m) - Int_x_ejkR_RXw_neg
            
            # 计算电标位
            Phi_Inter_pos = np.divide(Int_ejkR_R_pos,l_m)
            Phi_Inter_neg = np.divide(Int_ejkR_R_neg,l_m)
            Phi_Inert_pos_Full = Phi_Inter_pos[:,Seg_Pos]-Phi_Inter_pos[:,Seg_Neg]
            Phi_Inert_neg_Full = Phi_Inter_neg[:,Seg_Pos]-Phi_Inter_neg[:,Seg_Neg]
            Phi_Inert_pos_Full = np.divide(Phi_Inert_pos_Full, ln_pos) # 乘以外面测试函数
            Phi_Inert_neg_Full = np.divide(Phi_Inert_neg_Full, ln_neg) # 乘以外面测试函数
            Phi_Inert_pos_Full = np.divide(Phi_Inert_pos_Full, wavenumber**2) # 除以k**
            Phi_Inert_neg_Full = np.divide(Phi_Inert_neg_Full, wavenumber**2) # 除以k**
            # 计算磁矢量位
            hl_n_Dot_hlm_pos = np.sum(np.multiply(hln_pos.reshape([3,1]),hl_m),axis=0)
            hl_n_Dot_hlm_neg = np.sum(np.multiply(hln_neg.reshape([3,1]),hl_m),axis=0)
            
            A_Inter_Pos = np.divide(Int_x_ejkR_RXw_pos,l_m)
            A_Inter_Pos = np.multiply(A_Inter_Pos,hl_n_Dot_hlm_pos)
            A_Inter_Neg = np.divide(Int_l_x_ejkR_RXw_pos,l_m)
            A_Inter_Neg = np.multiply(A_Inter_Neg,hl_n_Dot_hlm_pos)
            A_Inter_pos_Full = A_Inter_Pos[:,Seg_Pos]+A_Inter_Neg[:,Seg_Neg]
            A_Inter_pos_Full = np.multiply(A_Inter_pos_Full,0.5)# 乘以外部的测试函数
            
            A_Inter_Pos = np.divide(Int_x_ejkR_RXw_neg,l_m)
            A_Inter_Pos = np.multiply(A_Inter_Pos,hl_n_Dot_hlm_neg)
            A_Inter_Neg = np.divide(Int_l_x_ejkR_RXw_neg,l_m)
            A_Inter_Neg = np.multiply(A_Inter_Neg,hl_n_Dot_hlm_neg)
            A_Inter_neg_Full = A_Inter_Pos[:,Seg_Pos]+A_Inter_Neg[:,Seg_Neg]
            A_Inter_neg_Full = np.multiply(A_Inter_neg_Full,0.5)# 乘以外部的测试函数
            
            Z[n,:]= np.multiply(np.subtract(A_Inter_pos_Full, Phi_Inert_pos_Full), ln_pos*an_pos)
            Z[n,:] = Z[n,:] + np.multiply(np.add(A_Inter_neg_Full, Phi_Inert_neg_Full), ln_neg*an_neg)            
        except ValueError as ve:
            print ve
            raise
        except IndexError as ie:
            print ie
            raise
        except Exception as e:
            print e
            raise
    return Z      
# In[117]:

def VecFilling(rooftop_attached):    
    # 循环迭代 产生右端向量和左端加载量
    try:
        V = np.zeros((len(rooftop_attached),1),dtype=np.complex) #右端向量
    except Exception as e:
        print e
        raise    
    try:
        Loads = np.zeros((len(rooftop_attached),1),dtype=np.complex) #左端加载量
    except Exception as e:
        print e
        raise    
    for n,rooftop_n in enumerate(rooftop_attached): # 遍历所有测试
        try:
            # 判断是否是加载电压源的单元
            if rooftop_n['Port'][0] == True and rooftop_n['V_e'][0] == True: # 是
                temp = scipy.exp(-1.j*rooftop_n['V_e'][2])*rooftop_n['V_e'][1] # 根据指定相位差计算右端 step1
#                V[n] = temp # step 2
                if rooftops[n]['pos'][0]!=-1:
                    width = mesh[rooftops[n]['pos'][0]][3]
                else:
                    width = mesh[rooftops[n]['neg'][0]][3]
                V[n] = temp*_jOmegaMu*pi4*width 
            else: # 不是
                V[n] = float(0) # 指定为右端项为0
        except Exception as e:
            print e
            raise
        try:
            # 判断是否有加载阻抗的单元
            if rooftop_n['Port'][0] == True: # 是
                temp = rooftop_n['Port'][1]
                if rooftops[n]['pos'][0]!=-1:
                    width = mesh[rooftops[n]['pos'][0]][3]
                else:
                    width = mesh[rooftops[n]['neg'][0]][3]
                
                Loads[n] = temp*_jOmegaMu*pi4*width**2 # step 2 乘以width^2是要进行集总和分布的转换
            else: #
                Loads[n] = float(0) # 指定为右端项为0
        except Exception as e:
            print e
            raise
    return (V,Loads)


# In[119]:
def getI(V, Icoef, rooftop_attachedf,rooftops, mesh):
    
    # 循环迭代 产生右端向量和左端加载量
    Vmatrix = list()
    Imatrix = list()
    
    for n,rooftop_n in enumerate(rooftop_attached): # 遍历所有测试
        try:
            # 判断是否是加载电压源的单元
            if rooftop_n['Port'][0] == True and rooftop_n['V_e'][0] == True: # 是
                if rooftops[n]['pos'][0]!=-1:
                    width = mesh[rooftops[n]['pos'][0]][3]
                else:
                    width = mesh[rooftops[n]['neg'][0]][3]
                Vmatrix.append(V[n]/_jOmegaMu/pi4/width)
                
                Icell = Icoef[n]*width
                Imatrix.append(Icell)
            else: # 不是
                pass
        except Exception as e:
            print e
            raise
    return (np.array(Vmatrix),np.array(Imatrix))

# In[120]:
def  genRadiationPattern2(ths_phs):  
    try:
        rr = 100.0
        # 对方向角度中的集合进行迭代
        brs = np.zeros([len(ths_phs),3])
        for iid,th_ph in enumerate(ths_phs):
    
            th0,ph0 = th_ph # 俯仰角和方位角
            # 根据俯仰角和方位角计算三个向量
            try:
                th0,ph0 = th_ph # 俯仰角和方位角
                # 根据俯仰角和方位角计算三个向量
                if th0 >=0 and th0 <= scipy.pi:
                    br = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*rr # 1.j*k向量
                elif th0 < 0 and th0 >= -scipy.pi:
                    th0 = -th0
                    ph0 = ph0 +scipy.pi
                    br = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*rr # 1.j*k向量
                elif th0 > scipy.pi and th0 <= scipy.pi*2.0:
                    th0 = th0-scipy.pi
                    ph0 = ph0 +scipy.pi                
                    br = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*rr # 1.j*k向量
            except Exception as e:
                print e
                raise
            brs[iid] = br 
        
        brs = brs.transpose()
        
        # 正矩形的编号
        seg_m_id_pos = [edge_m['pos'][0] for edge_m in rooftops]
        mesh_seg_m_id_pos = [mesh[tt] for tt in seg_m_id_pos]
        # 正矩形的宽
        am_pos = np.array( [seg_m_pos[3] for seg_m_pos in  mesh_seg_m_id_pos] )
        # 矩形的中心
        rmc_pos = np.array( [seg_m_pos[1]+seg_m_pos[0] for seg_m_pos in  mesh_seg_m_id_pos] )
        rmc_pos = np.multiply(0.5, rmc_pos)
        rmc_pos = rmc_pos.transpose()
        # 单极子的矩量
        bm_pos = np.array( [seg_m_pos[1]-seg_m_pos[0] for seg_m_pos in  mesh_seg_m_id_pos] )
        bm_pos = np.multiply(0.5, bm_pos)
        bm_pos = bm_pos.transpose()     
        bm_pos = np.multiply(np.multiply(bm_pos,am_pos),Icoef.reshape(Icoef.shape[0]))
        # 单极子的距离
        br_rmc_pos = np.array([brs[:,ii:ii+1]-rmc_pos for ii in xrange(brs.shape[-1])])
        r_len = np.sqrt( np.sum(br_rmc_pos**2,axis=1) )
        # 计算Ｅ
        C = 1/r_len**2*(1-1j/(wavenumber*r_len))
        ejkr = np.exp(-1j*wavenumber*r_len)
        temp1 = np.sum(np.multiply(bm_pos,br_rmc_pos),axis=1)/r_len**2
        temp2 = np.multiply( temp1.transpose([1,0]), br_rmc_pos.transpose([1,2,0]) )
        bM = temp2.transpose([2,0,1])
        aita = 377.0 
        
        tempEE1 = aita/pi4*ejkr*(1j*wavenumber/r_len+C)
        tempEE2 = aita/pi4*ejkr*2.*C
        tempEE3 = bM-bm_pos
        Es_pos = np.array([ tempEE1[ii,:]*tempEE3[ii,:,:]+tempEE2[ii,:]*bM[ii,:,:]  for ii in xrange(tempEE2.shape[0])])
        # 计算H
        temp3 = [np.cross(bm_pos.transpose(),br_rmc_pos[ii,:,:].transpose()).transpose() for ii in xrange(br_rmc_pos.shape[0])]
        temp3 = np.array(temp3)
        Hs_pos = np.array([temp3[ii,:,:]*C[ii,:]*ejkr[ii,:]*1j*wavenumber/pi4 for ii in xrange(temp3.shape[0])])
        
        # －矩形的编号
        seg_m_id_neg = [edge_m['neg'][0] for edge_m in rooftops]
        mesh_seg_m_id_neg = [mesh[tt] for tt in seg_m_id_neg]
        # －矩形的宽
        am_neg = np.array( [seg_m_neg[3] for seg_m_neg in  mesh_seg_m_id_neg] )
        # 矩形的中心
        rmc_neg = np.array( [seg_m_neg[1]+seg_m_neg[0] for seg_m_neg in  mesh_seg_m_id_neg] )
        rmc_neg = np.multiply(0.5, rmc_neg)
        rmc_neg = rmc_neg.transpose()   
        # 单极子的矩量
        bm_neg = np.array( [seg_m_neg[1]-seg_m_neg[0] for seg_m_neg in  mesh_seg_m_id_neg] )
        bm_neg = np.multiply(0.5, bm_neg)
        bm_neg = bm_neg.transpose()   
        bm_neg = np.multiply(np.multiply(bm_neg,am_neg),Icoef.reshape(Icoef.shape[0]))
        # 单极子的距离
        br_rmc_neg = np.array([brs[:,ii:ii+1]-rmc_neg for ii in xrange(brs.shape[-1])])
        r_len = np.sqrt( np.sum(br_rmc_neg**2,axis=1) )
        # 计算Ｅ
        C = 1/r_len**2*(1-1j/(wavenumber*r_len))
        ejkr = np.exp(-1j*wavenumber*r_len)
        temp1 = np.sum(np.multiply(bm_neg,br_rmc_neg),axis=1)/r_len**2
        temp2 = np.multiply( temp1.transpose([1,0]), br_rmc_neg.transpose([1,2,0]) )
        bM = temp2.transpose([2,0,1])
        
        tempEE1 = aita/pi4*ejkr*(1j*wavenumber/r_len+C)
        tempEE2 = aita/pi4*ejkr*2.*C
        tempEE3 = bM-bm_neg
        Es_neg = np.array([ tempEE1[ii,:]*tempEE3[ii,:,:]+tempEE2[ii,:]*bM[ii,:,:]  for ii in xrange(tempEE2.shape[0])])
        # 计算H
        temp3 = [np.cross(bm_neg.transpose(),br_rmc_neg[ii,:,:].transpose()).transpose() for ii in xrange(br_rmc_neg.shape[0])]
        temp3 = np.array(temp3)
        Hs_neg = np.array([temp3[ii,:,:]*C[ii,:]*ejkr[ii,:]*1j*wavenumber/pi4 for ii in xrange(temp3.shape[0])])

        # 场的叠加
        Etotal = np.sum(Es_pos+Es_neg,axis=2)
        Htotal = np.sum(Hs_pos+Hs_neg,axis=2)
        # 计算波印廷矢量
        WWtotal = np.real(np.cross(Etotal,np.conjugate(Htotal)))*0.5
        # 计算功率密度
        RPs = np.multiply(WWtotal.transpose(),brs)
        RPs = np.sum(RPs,axis=0)*rr
        return RPs
    except Exception as ve:
        print ve
        raise


 # In[106]:
def calGain():
    try:
        ths = np.linspace(0,np.pi,90)
        phs = np.linspace(0,np.pi*2,180)
        xx,yy = np.meshgrid(ths,phs)
        
        x1 = xx.reshape(len(ths)*len(phs))
        y1 = yy.reshape(len(ths)*len(phs))
        ths_phs = zip(x1,y1)
        
        augs = genRadiationPattern2(ths_phs)
        augs = augs.reshape(xx.shape)

        max_d = np.max(augs)
        
        return (max_d,[ths,phs,augs])
    except Exception as e:
        print e
        print x1
        print y1
        raise

 # In[106]:       
if __name__ == '__main__':
    # In[106]:
    import   mesh0
#    import mesh3 as mesh0
#    import mesh_circle as mesh0
    
    mesh = mesh0.mesh2
    rooftops = mesh0.rooftops
    rooftop_attached = mesh0.rooftop_attached
    
    # In[106]:
    # 设置基本参数
    freq = 300e6 #设置频率
    
    from scipy import constants as C
    
    epsi = C.epsilon_0 #设置介电常数
    mu = C.mu_0 # 设置磁导率
    vec = 1./np.sqrt(epsi*mu) # 计算光速
        
    circleFreq = 2*np.pi*freq # 计算圆频率
    wavelength = vec/freq # 计算波长
    wavenumber = 2*np.pi/wavelength # 计算波数
    
    pi4 = np.pi*4. #预先计算公共因子 
    jOmegaMuPi4 = circleFreq*mu/pi4*1.j #预先计算公共因子 
    _jOmegaEpsPi4 = -1.j/circleFreq/epsi/pi4 # 预先计算公共因子
    _jOmegaMu = -1.j/circleFreq/mu # 预先计算公共因子
    jK = 1.j*wavenumber # 预先计算公共因子

    
    # In[106]:   
    t_matrix_filling = time.clock()

#    Z = MatrixFilling_opt(mesh,rooftops)
    Z = MatrixFilling_opt_new(mesh,rooftops)

    print "matrix filling completed"  
    print "%.3f "%(time.clock()-t_matrix_filling), "secondes passed"
    print "=="*30
     
    # In[106]:   
    t_vector_filling = time.clock()
    V,Loads = VecFilling(rooftop_attached)
    
    print "vector filling completed"  
    print "%.3f "%(time.clock()-t_vector_filling), "secondes passed"
    print "=="*30
   
    # In[118]:
    import scipy.linalg
    try:
        t_solving = time.clock()
        Mat = Z - np.diag(Loads[:,0])
        Icoef = scipy.linalg.solve(Mat,V)
        
        Is = np.zeros((len(mesh),1),dtype=np.complex) # 每个线段上的电流
        for iidd, func in enumerate(rooftops):# 遍历所有的电流系数
            # 存入相应的线段
            if func['pos'][0] == -1:
                pass
            else:
                Is[func['pos'][0]] = Is[func['pos'][0]]+Icoef[iidd]
            if func['neg'][0] == -1:
                pass
            else:
                Is[func['neg'][0]] = Is[func['neg'][0]]+Icoef[iidd]
        # 将电流系数乘以线段的宽度
        for iid, meshcell in enumerate(mesh):
            Is[iid] = Is[iid]*meshcell[3]
    
        Is = Is*0.5# 将线段的两个端点的电流系数进行平均，就能得到该线段的中间点的电流
        
    except Exception as e:
        print e


    '''
    # 将网格进行绘制
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D
    print "current (phase)"
    fig = plt.figure()
    plt.plot(xrange(len(Is)),scipy.angle(Is)*180/scipy.pi)
    plt.show()
    
    print "current (augment)"
    fig = plt.figure()
    plt.plot(xrange(len(Is)),scipy.absolute(Is))
    plt.show()
    
    print "current (real)"
    fig = plt.figure()
    plt.plot(xrange(len(Is)),scipy.real(Is))
    plt.show()
    
    print "current (imag)"
    fig = plt.figure()
    plt.plot(xrange(len(Is)),scipy.imag(Is))
    plt.show()
    '''
    
    V_port,I_port = getI(V, Icoef, rooftop_attached,rooftops, mesh)
    
    print "solving completed"  
    print "%.3f "%(time.clock()-t_solving), "secondes passed"
    
    print "V_port = "
    print V_port
    print "I_port = "
    print I_port
    
    print "=="*30
    
    # In[]    

    t_cal_gain = time.clock()
    direct = calGain()

    D0 = np.max(direct[1][2])/scipy.real(scipy.sum(scipy.conjugate(V_port)*I_port))*2*pi4
    
    print "gain obtained"
    print "%.3f "%(time.clock()-t_cal_gain), "secondes passed"
    
    print "direct = ", D0
    print "gain = ", 10.*np.log10(D0), 'dBi'
    print "=="*30


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    ths,phs,aug = direct[1]
    ts,ps = np.meshgrid(ths,phs)
            
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    surf = ax1.plot_surface(ts,ps,aug, linewidth=0)
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.show()

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    xs = aug*np.sin(ts)*np.cos(ps)/np.max(aug)
    ys = aug*np.sin(ts)*np.sin(ps)/np.max(aug)
    zs = aug*np.cos(ts)/np.max(aug)
    surf = ax2.plot_surface(xs,ys,zs, linewidth=0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print "=="*30


    # In[119]:

    # 指定方向角集合
    ths = np.linspace(-np.pi,np.pi,720)
    ths_phs = [(xx,np.pi*0) for xx in ths]
    
    t_cal_gain2 = time.clock()        
    RPs = genRadiationPattern2(ths_phs)
    
    print "gain (2) obtained"    
    print "%.3f "%(time.clock()-t_cal_gain2), "secondes passed"
    
    import matplotlib.pylab as plt
        
    try:
        aug_rp = RPs
        max_rp = np.max(aug_rp)
        print "total"
        if max_rp != 0: 
            print "relative value"
            fig = plt.figure()
            rrs = aug_rp
            rrs = 10.*np.log10(aug_rp/max_rp)+40.
            for iirrs in xrange(len(rrs)):
                if rrs[iirrs] < 0:
                    rrs[iirrs] = 0.
            plt.polar(ths,rrs)
            plt.show()
        else:
            print "absolute value"
            fig = plt.figure()
            plt.polar(ths,aug_rp)
            plt.show()
    except Exception as e:
        print e
        
    print "=="*30