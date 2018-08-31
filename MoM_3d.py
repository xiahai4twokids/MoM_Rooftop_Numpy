
# coding: utf-8
import numpy as np
import scipy

import utilities_3d as utilities



#print wavenumber
# In[114]:
# 循环迭代 产生矩阵
# 下面的迭代适用3Ｄ问题

def MatrixFilling(mesh,rooftops):
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
            rn1_pos = seg_n_pos[0]
            an_pos = seg_n_pos[3] #宽度
            ln_pos = seg_n_pos[2] # 线段的长度
            hln_pos = (seg_n_pos[1]-seg_n_pos[0])/ln_pos # 计算测试的向量
            han_pos = seg_n_pos[4]   
        except Exception as e:
            print e
            raise   
        # 计算neg线段的测试
        try:
            seg_n_id_neg = edge_n['neg'][0]
            seg_n_neg = mesh[seg_n_id_neg]
            rn1_neg = seg_n_neg[0]
            an_neg = seg_n_neg[3] #宽度
            ln_neg = seg_n_neg[2] # 长度
            hln_neg = (seg_n_neg[1]-seg_n_neg[0])/ln_neg # 计算测试的向量
            han_neg = seg_n_neg[4]
        except Exception as e:
            print e
            raise 
            
        for m, edge_m in enumerate(rooftops):
            tempA = scipy.zeros(4,dtype=scipy.complex64)
            tempPhi=scipy.zeros(4,dtype=scipy.complex64)
            
            # 计算pos线段的展开
            try:
                seg_m_id_pos = edge_m['pos'][0]
                seg_m_pos = mesh[seg_m_id_pos]
                rm1_pos = seg_m_pos[0]
                am_pos = seg_m_pos[3] #宽度
                lm_pos = seg_m_pos[2] # 长度
                hlm_pos = (seg_m_pos[1]-seg_m_pos[0])/lm_pos # 计算测试的向量
                ham_pos = seg_m_pos[4]
            except Exception as e:
                print e
                raise
            # 计算neg线段的展开
            try:
                seg_m_id_neg = edge_m['neg'][0]
                seg_m_neg = mesh[seg_m_id_neg]
                rm1_neg = seg_m_neg[0]
                am_neg = seg_m_neg[3] #偏置的长度
                lm_neg = seg_m_neg[2] # 线段的长度
                hlm_neg = (seg_m_neg[1]-seg_m_neg[0])/lm_neg # 计算测试的向量
                ham_neg = seg_m_neg[4]
            except Exception as e:
                print e
                raise
                
            try:
                tempA[0] = utilities.A_outer_quad_global(0,'pos','pos',
                     rn1_pos,hln_pos,han_pos,ln_pos,an_pos,
                     rm1_pos,hlm_pos,ham_pos,lm_pos,am_pos,
                     wavenumber)
                tempPhi[0] = utilities.Phi_outer_quad_global(0,'pos','pos',
                       rn1_pos,hln_pos,han_pos,ln_pos,an_pos,
                       rm1_pos,hlm_pos,ham_pos,lm_pos,am_pos,
                       wavenumber)
            except Exception as e:
                print e
                raise
            try:
                tempA[1] = utilities.A_outer_quad_global(0,'pos','neg',
                     rn1_pos,hln_pos,han_pos,ln_pos,an_pos,
                     rm1_neg,hlm_neg,ham_neg,lm_neg,am_neg,
                     wavenumber)
                tempPhi[1] = utilities.Phi_outer_quad_global(0,'pos','neg',
                       rn1_pos,hln_pos,han_pos,ln_pos,an_pos,
                       rm1_neg,hlm_neg,ham_neg,lm_neg,am_neg,
                       wavenumber)
            except Exception as e:
                print e
                raise
        
            try:
                tempA[2] = utilities.A_outer_quad_global(0,'neg','pos',
                     rn1_neg,hln_neg,han_neg,ln_neg,an_neg,
                     rm1_pos,hlm_pos,ham_pos,lm_pos,am_pos,
                     wavenumber)
                tempPhi[2] =  utilities.Phi_outer_quad_global(0,'neg','pos',
                       rn1_neg,hln_neg,han_neg,ln_neg,an_neg,
                       rm1_pos,hlm_pos,ham_pos,lm_pos,am_pos,
                       wavenumber)
            except Exception as e:
                print e
                raise

            try:
                tempA[3] = utilities.A_outer_quad_global(0,'neg','neg',
                     rn1_neg,hln_neg,han_neg,ln_neg,an_neg,
                     rm1_neg,hlm_neg,ham_neg,lm_neg,am_neg,
                     wavenumber)
                tempPhi[3] =  utilities.Phi_outer_quad_global(0,'neg','neg',
                       rn1_neg,hln_neg,han_neg,ln_neg,an_neg,
                       rm1_neg,hlm_neg,ham_neg,lm_neg,am_neg,
                       wavenumber)
            except Exception as e:
                print e
                raise
            # 将四个进行相加 
            try:
                Z[n][m] = scipy.sum(tempA)-scipy.sum(tempPhi)/wavenumber**2
            except Exception as e:
                raise
    return Z
    
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
        W_x,P_x = utilities.getWeightAndPoint_S(1)
        W_y,P_y = utilities.getWeightAndPoint_S(0)
        # 存储每个矩形上的高斯点 [高斯点个数，坐标，区间段]
        PP_x, PP_y = np.meshgrid(P_x,P_y)
        WW_x, WW_y = np.meshgrid(W_x,W_y)
        shift_x = np.array([np.multiply(hl_m[:,ii:ii+1],np.multiply(l_m[ii:ii+1],(PP_x+1.)*0.5)) for ii in xrange(hl_m.shape[1])])
        shift_y = np.array([np.multiply(ha_m[:,ii:ii+1],np.multiply(a_m[ii:ii+1],PP_y*0.5)) for ii in xrange(hl_m.shape[1])])        
        Points = np.add(x0_m,  (shift_x + shift_y).transpose([2,1,0]))
        # 存放＋矩形的编号
        Seg_Pos = np.array([rooftopcell['pos'][0] for rooftopcell in rooftops])
        # 存放——矩形的编号
        Seg_Neg = np.array([rooftopcell['neg'][0] for rooftopcell in rooftops])
                 
    except ValueError as ve:
        print ve
    except Exception as e:
        print e
            
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
            print Z
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
from numba import jit

def calE_cell(br,bm):
    r_len = np.sqrt(np.sum(br**2))
    C = 1/r_len**2*(1-1.j/(wavenumber*r_len))
    ejkr = scipy.exp(-1.j*wavenumber*r_len)
    bM = np.dot(br,bm)/r_len**2*br
    aita = 377.0
    return aita/pi4*ejkr*(1.j*wavenumber/r_len+C)*(bM-bm)+aita/pi4*ejkr*( 2.*C*bM )  
    pass

def calH_cell(br,bm):
    r_len = np.sqrt(np.sum(br**2))
    C = 1/r_len**2*(1.-1.j/(wavenumber*r_len))
    ejkr = scipy.exp(-1.j*wavenumber*r_len)
    return np.cross(bm,br)*C*ejkr*1.j*wavenumber/pi4
    pass

def calE_total(br):
    try:
        temp4E_x = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)
        temp4E_y = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)
        temp4E_z = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)

        for m, edge_m in enumerate(rooftops):
            # 计算pos线段的展开
            try:
                seg_m_id_pos = edge_m['pos'][0]
                seg_m_pos = mesh[seg_m_id_pos]
                am_pos = seg_m_pos[3] #宽度
                bm_pos = (seg_m_pos[1]-seg_m_pos[0])*0.5*am_pos*Icoef[m] # 计算测试的矩
                rmc_pos = (seg_m_pos[1]+seg_m_pos[0])*0.5
                temp_pos = calE_cell(br-rmc_pos,bm_pos)
                temp4E_x[m] =   temp_pos[0]
                temp4E_y[m] =   temp_pos[1]
                temp4E_z[m] =   temp_pos[2]
#                raise Exception
            except Exception as e:
                print e
                print temp_pos
                print bm_pos
                print br-rmc_pos
                raise

            # 计算neg线段的展开
            try:
                seg_m_id_neg = edge_m['neg'][0]
                seg_m_neg = mesh[seg_m_id_neg]
                am_neg = seg_m_neg[3] #偏置的长度
                bm_neg = (seg_m_neg[1]-seg_m_neg[0])*0.5*am_neg*Icoef[m] # 计算测试的向量
                rmc_neg = (seg_m_neg[1]+seg_m_neg[0])*0.5
                temp_neg = calE_cell(br-rmc_neg,bm_neg)
                temp4E_x[m] = temp4E_x[m] +  temp_neg[0]
                temp4E_y[m] = temp4E_y[m] +  temp_neg[1]
                temp4E_z[m] = temp4E_z[m] +  temp_neg[2]
            except Exception as e:
                print e
                raise
#            print temp4E_x[m]
#            print temp4E_y[m]
#            print temp4E_z[m]
        return scipy.array([scipy.sum(temp4E_x),scipy.sum(temp4E_y),scipy.sum(temp4E_z)])
        pass
    except Exception as e:
        print e
        print scipy.shape(temp4E_x)
        raise


def calH_total(br):
    temp4E_x = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)
    temp4E_y = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)
    temp4E_z = scipy.zeros(scipy.shape(Icoef),dtype=scipy.complex64)
    for m, edge_m in enumerate(rooftops):
        # 计算pos线段的展开
        try:
            seg_m_id_pos = edge_m['pos'][0]
            seg_m_pos = mesh[seg_m_id_pos]
            am_pos = seg_m_pos[3] #宽度
            bm_pos = (seg_m_pos[1]-seg_m_pos[0])*0.5*am_pos*Icoef[m] # 计算测试的矩
            rmc_pos = (seg_m_pos[1]+seg_m_pos[0])*0.5
            temp_pos = calH_cell(br-rmc_pos,bm_pos)
            temp4E_x[m] =  temp_pos[0]
            temp4E_y[m] =  temp_pos[1]
            temp4E_z[m] =  temp_pos[2]
        except Exception as e:
            print e
            raise
        # 计算neg线段的展开
        try:
            seg_m_id_neg = edge_m['neg'][0]
            seg_m_neg = mesh[seg_m_id_neg]
            am_neg = seg_m_neg[3] #偏置的长度
            bm_neg = (seg_m_neg[1]-seg_m_neg[0])*0.5*am_neg*Icoef[m] # 计算测试的向量
            rmc_neg = (seg_m_neg[1]+seg_m_neg[0])*0.5
            temp_neg = calH_cell(br-rmc_neg,bm_neg)
            temp4E_x[m] = temp4E_x[m] +  temp_neg[0]
            temp4E_y[m] = temp4E_y[m] +  temp_neg[1]
            temp4E_z[m] = temp4E_z[m] +  temp_neg[2]
        except Exception as e:
            print e
            raise
        
    return scipy.array([scipy.sum(temp4E_x),scipy.sum(temp4E_y),scipy.sum(temp4E_z)])    

def  genRadiationPattern2(ths_phs):  
    try:
#        print ths_phs
        RPs = [0 for _ in ths_phs]
        # 定义一个辅助空间保存各个单元的因子，其作用是利用向量乘积的办法计算远处的电场
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
        EEs = scipy.array([calE_total(br_cell) for br_cell in brs])
        HHs = scipy.array([calH_total(br_cell) for br_cell in brs])
        WWs = scipy.array([scipy.real(scipy.cross(EE,scipy.conjugate(HH))) for EE,HH in zip(EEs,HHs)])*0.5
        RPs = scipy.array([rr*np.dot(WW,br_cell) for WW,br_cell in zip(WWs,brs)])
    #    for br in brs:
    #        # 计算电场    
    #        EE = calE_total(br)
    #        #　计算磁场
    #        HH = calH_total(br)
    #        # 计算坡印廷矢量
    #        WW = scipy.real(scipy.cross(EE,scipy.conjugate(HH)))*0.5
    #        # 计算功率密度
    #        UU = rr*np.dot(WW,br)
    #        # 保存，并返回
    #        RPs[iid] = UU
      
        return RPs
    except Exception as e:
        print e
        print ths_phs
        print np.shape(brs)
        print br
        raise

    
# 定义与集合相应的方向图
def genRadiationPattern(ths_phs):
    RPs_th = [0 for _ in ths_phs]
    RPs_ph = [0 for _ in ths_phs]

    # 定义一个辅助空间保存各个单元的因子，其作用是利用向量乘积的办法计算远处的电场
    try:
        RP_sep_th = np.zeros((len(mesh),1),dtype=complex) # 定义空间，并初始化为零
        RP_sep_ph = np.zeros((len(mesh),1),dtype=complex) # 定义空间，并初始化为零
    except Exception as e:
        print e
        raise   

    # 对方向角度中的集合进行迭代
    for iid,th_ph in enumerate(ths_phs):

        th0,ph0 = th_ph # 俯仰角和方位角
        # 根据俯仰角和方位角计算三个向量
        try:
            th0,ph0 = th_ph # 俯仰角和方位角
            # 根据俯仰角和方位角计算三个向量
            if th0 >=0 and th0 <= scipy.pi:
                jk_bar = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*wavenumber*1.j # 1.j*k向量
                th_hat = np.array([scipy.cos(th0)*scipy.cos(ph0),scipy.cos(th0)*scipy.sin(ph0),-scipy.sin(th0)]) # theta分量
                ph_hat = np.array([-scipy.sin(ph0),scipy.cos(ph0),0]) # phi分量
            elif th0 < 0 and th0 >= -scipy.pi:
                th0 = -th0
                ph0 = ph0 + +scipy.pi
                jk_bar = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*wavenumber*1.j # 1.j*k向量
                th_hat = np.array([scipy.cos(th0)*scipy.cos(ph0),scipy.cos(th0)*scipy.sin(ph0),-scipy.sin(th0)]) # theta分量
                ph_hat = np.array([-scipy.sin(ph0),scipy.cos(ph0),0]) # phi分量
            elif th0 > scipy.pi and th0 <= scipy.pi*2.0:
                th0 = th0-scipy.pi
                ph0 = ph0 + +scipy.pi                
                jk_bar = np.array([scipy.sin(th0)*scipy.cos(ph0),scipy.sin(th0)*scipy.sin(ph0),scipy.cos(th0)])*wavenumber*1.j # 1.j*k向量
                th_hat = np.array([scipy.cos(th0)*scipy.cos(ph0),scipy.cos(th0)*scipy.sin(ph0),-scipy.sin(th0)]) # theta分量
                ph_hat = np.array([-scipy.sin(ph0),scipy.cos(ph0),0]) # phi分量  
        except Exception as e:
            print e
            raise
        # 对网格进行迭代
        for n,seg_n in enumerate(mesh): 
            try:
                r_nc = (seg_n[0]+seg_n[1])/2. # 单元的中间点
                temp_e = scipy.exp( scipy.dot(jk_bar,r_nc) ) # 方向图step1
                temp_th = scipy.dot((seg_n[1]-seg_n[0]),th_hat) # 方向图step2
                temp_ph = scipy.dot((seg_n[1]-seg_n[0]),ph_hat) # 方向图step2
                if scipy.absolute(temp_th)<1.e-10: 
                    RP_sep_th[n] = 0. # 方向图step3
                else:
                    RP_sep_th[n] = temp_th*temp_e # 方向图step3
                if scipy.absolute(temp_ph)<1.e-10: 
                    RP_sep_ph[n] = 0. # 方向图step3
                else:
                    RP_sep_ph[n] = temp_ph*temp_e # 方向图step3
    #             print RP_sep[n]
            except Exception as e:
                print e
                raise
        try:         
            RPs_th[iid] = scipy.sum(scipy.dot(Is[:,0],RP_sep_th)) # 将各个单元的方向图进行综合
            RPs_ph[iid] = scipy.sum(scipy.dot(Is[:,0],RP_sep_ph)) # 将各个单元的方向图进行综合
        except Exception as e:
            print e
            raise
    return [RPs_th,RPs_ph]
 # In[106]:
def calGain():
    try:
        ths = np.linspace(0,np.pi,60)
#        w_ths = np.sin(ths)*(np.pi/len(ths))
        phs = np.linspace(0,np.pi*2,120,endpoint=False)
#        w_phs = np.pi*2./len(phs)*np.ones(np.shape(phs))
        xx,yy = np.meshgrid(ths,phs)
#        augs = np.empty(np.shape(xx))
#        weights = np.empty(np.shape(xx))
        ths_phs_fun = lambda ii,jj: [(xx[ii][jj],yy[ii][jj]),]
        augs = np.array([[genRadiationPattern2(ths_phs_fun(ii,jj))[0] for jj in xrange(len(xx[0]))] for ii in xrange(len(xx))])
#        for ii in xrange(len(xx)):
#            for jj in xrange(len(xx[ii])):
#                ths_phs = [(xx[ii][jj],yy[ii][jj]),]
#                augs[ii][jj] = genRadiationPattern2(ths_phs)[0]
#                augs[ii][jj] = genRadiationPattern2(ths_phs_fun(ii,jj))[0]
#                weights[ii][jj] = w_ths[ii]*w_phs[jj]
        max_d = np.max(augs)
#        direct = max_d*pi4/np.sum(augs*weights)
        direct = max_d
        
        return (direct,[ths,phs,augs])
    except Exception as e:
        print e
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
    Z = MatrixFilling_opt(mesh,rooftops)
#    print Z

    print "filling completed"   
    # In[106]:    
    V,Loads = VecFilling(rooftop_attached)
   
    # In[118]:
    import scipy.linalg
    try:
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


    # In[119]:
    
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

    V_port,I_port = getI(V, Icoef, rooftop_attached,rooftops, mesh)
    print "V_port = "
    print V_port
    print "I_port = "
    print I_port
    
    # In[]
    
    direct = calGain()
    D0 = np.max(direct[1][2])/scipy.real(scipy.sum(scipy.conjugate(V_port)*I_port))*2*pi4
    print "direct = ", D0
    print "gain = ", 10.*np.log10(D0), 'dBi'
    
    # In[119]:

    # 指定方向角集合
    ths = np.linspace(-np.pi,np.pi,61)
    ths_phs = [(xx,np.pi*0.) for xx in ths]
                
    RPs = genRadiationPattern2(ths_phs)
    
    import matplotlib.pylab as plt
        
    try:
        aug_rp = RPs
        max_rp = np.max(aug_rp)
        print "total"
        if max_rp != 0: 
            print "relative value"
            fig = plt.figure()
            rrs = aug_rp
#            rrs = 10.*np.log10(aug_rp/max_rp)+40.
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
    

    # In[]

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    
    ths,phs,aug = direct[1]
    ts,ps = np.meshgrid(ths,phs)

    colortuple = ('y', 'b')
    colors = np.empty(ts.shape, dtype=str)
    for ii in xrange(len(aug)):
        for jj in xrange(len(aug[ii])):
            colors[ii, jj] = colortuple[(ii + jj) % len(colortuple)]
            
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
#    surf = ax.plot_surface(xs,ys,zs, facecolors=colors, linewidth=0)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
