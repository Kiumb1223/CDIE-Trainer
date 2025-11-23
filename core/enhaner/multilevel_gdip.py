import math 
import torch
import torchvision 

class VisionEncoder(torch.nn.Module):
    def __init__(self,encoder_output_dim=256,base_channel=32):
        super(VisionEncoder,self).__init__()
        print("VisionEncoder with 5 conv layers")
        # conv_1
        self.conv_1 = torch.nn.Sequential(torch.nn.Conv2d(3,64,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        self.max_pool_1 = torch.nn.AvgPool2d((3,3),(2,2))
        self.adp_pool_1 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_1 = torch.nn.Sequential(torch.nn.Linear(64,encoder_output_dim),
                                                torch.nn.ReLU(True))
        
        # conv_2
        self.conv_2 = torch.nn.Sequential(torch.nn.Conv2d(64,128,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        self.max_pool_2 = torch.nn.AvgPool2d((3,3),(2,2))
        self.adp_pool_2 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_2 = torch.nn.Sequential(torch.nn.Linear(128,encoder_output_dim),
                                                torch.nn.ReLU(True))
        # conv_3
        self.conv_3 = torch.nn.Sequential(torch.nn.Conv2d(128,256,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        self.max_pool_3 = torch.nn.AvgPool2d((3,3),(2,2))
        self.adp_pool_3 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_3 = torch.nn.Sequential(torch.nn.Linear(256,encoder_output_dim),
                                                torch.nn.ReLU(True))
        
        # conv_4
        self.conv_4 = torch.nn.Sequential(torch.nn.Conv2d(256,512,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        self.max_pool_4 = torch.nn.AvgPool2d((3,3),(2,2))
        self.adp_pool_4 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_4 = torch.nn.Sequential(torch.nn.Linear(512,encoder_output_dim),
                                                torch.nn.ReLU(True))
        
        # conv_5
        self.conv_5 = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        # self.max_pool_5 = torch.nn.AvgPool2d((3,3),(2,2))
        self.adp_pool_5 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_5 = torch.nn.Sequential(torch.nn.Linear(1024,encoder_output_dim),
                                                torch.nn.ReLU(True))
        
        # conv_6
        # self.conv_6 = torch.nn.Sequential(torch.nn.Conv2d(1024,2048,kernel_size = 3 , stride = 1),
        #                                 torch.nn.ReLU(True))
        # self.adp_pool_6 = torch.nn.AdaptiveAvgPool2d((1,1))
        # self.linear_proj_6 = torch.nn.Sequential(torch.nn.Linear(2048,encoder_output_dim),
        #                                         torch.nn.ReLU(True))
    def forward(self,x):
        out_x = self.conv_1(x)
        max_pool_1 = self.max_pool_1(out_x)
        adp_pool_1 = self.adp_pool_1(out_x)
        linear_proj_1 = self.linear_proj_1(adp_pool_1.view(adp_pool_1.shape[0],-1))
        
        out_x = self.conv_2(max_pool_1)
        max_pool_2 = self.max_pool_2(out_x)
        adp_pool_2 = self.adp_pool_2(out_x)
        linear_proj_2 = self.linear_proj_2(adp_pool_2.view(adp_pool_2.shape[0],-1))
        
        out_x = self.conv_3(max_pool_2)
        max_pool_3 = self.max_pool_3(out_x)
        adp_pool_3 = self.adp_pool_3(out_x)
        linear_proj_3 = self.linear_proj_3(adp_pool_3.view(adp_pool_3.shape[0],-1))
        
        out_x = self.conv_4(max_pool_3)
        max_pool_4 = self.max_pool_4(out_x)
        adp_pool_4 = self.adp_pool_4(out_x)
        linear_proj_4 = self.linear_proj_4(adp_pool_4.view(adp_pool_4.shape[0],-1))
        
        out_x = self.conv_5(max_pool_4)
        # max_pool_5 = self.max_pool_5(out_x)
        adp_pool_5 = self.adp_pool_5(out_x)
        linear_proj_5 = self.linear_proj_5(adp_pool_5.view(adp_pool_5.shape[0],-1))
        
        # out_x = self.conv_6(max_pool_5)
        # adp_pool_6 = self.adp_pool_6(out_x)
        # linear_proj_6 = self.linear_proj_6(adp_pool_6.view(adp_pool_6.shape[0],-1))

        return {'linear_proj_1':linear_proj_1,
                                'linear_proj_2':linear_proj_2,
                                'linear_proj_3':linear_proj_3,
                                'linear_proj_4':linear_proj_4,
                                'linear_proj_5':linear_proj_5}
                                #, 'linear_proj_6':linear_proj_6}

class GatedDIP(torch.nn.Module):
    def __init__(self,encoder_output_dim=256,num_of_gates=7):
        print("GDIP初始化")
        super(GatedDIP,self).__init__()
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))

        ##### GDIP
        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,num_of_gates,bias=True))
        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,3,bias=True))
        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Sharpning Module
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # De-Fogging Module
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,8,bias=True))

        self.bezier_module =torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,12,bias=True))

        self.kernel_size=3
        self.kernel_module =torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,6*self.kernel_size*self.kernel_size,bias=True))

    def tanh01(self,x : torch.tensor):
        """Shifts tanh from the [-1, 1] range to the [0, 1 range] and returns it for the given input. 

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Constrained tanh
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x : torch.tensor,left : float,right : float):
        """Returns tanh constrained to a particular range

        Args:
            x (torch.tensor): Input tensor
            left (float): Left bound 
            right (float): Right bound

        Returns:
            torch.tensor: Constrained tanh
        """
        return self.tanh01(x)*(right-left)+ left
    
    def rgb2lum(self,img: torch.tensor):
        """_summary_

        Args:
            img (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :,:] + 0.06 * img[:, 2, :, :]
        return img 
    
    def lerp(self ,a : int , b : int , l : torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b


    def dark_channel(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        z = x.min(dim=1)[0].unsqueeze(1)
        return z 
    
    def atmospheric_light(self,x : torch.tensor,dark : torch.tensor ,top_k : int=1000):
        """_summary_

        Args:
            x (torch.tensor): _description_
            top_k (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        h,w = x.shape[2],x.shape[3]
        imsz = h * w 
        numpx = int(max(math.floor(imsz/top_k),1))
        darkvec = dark.reshape(x.shape[0],imsz,1)
        imvec = x.reshape(x.shape[0],3,imsz).transpose(1,2)
        indices = darkvec.argsort(1)
        indices = indices[:,imsz-numpx:imsz]
        atmsum = torch.zeros([x.shape[0],1,3]).cuda()
        for b in range(x.shape[0]):
            for ind in range(1,numpx):
                atmsum[b,:,:] = atmsum[b,:,:] + imvec[b,indices[b,ind],:]
        a = atmsum/numpx
        a = a.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a
    
    def blur(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.gaussian_blur(x)


    def defog(self,x:torch.tensor ,latent_out : torch.tensor ,fog_gate : torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM 
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega,torch.tensor(0.1),torch.tensor(1.))
        dark_i = self.dark_channel(x) 
        a = self.atmospheric_light(x,dark_i)
        i = x/a 
        i = self.dark_channel(i)
        t = 1. - (omega*i)
        j = ((x-a)/(torch.maximum(t,torch.tensor(0.01))))+a
        j = (j - j.min())/(j.max()-j.min())
        j = j* fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
    
    def white_balance(self,x : torch.tensor,latent_out : torch.tensor ,wb_gate: torch.tensor):
        """ White balance of the image is predicted using latent output of an encoder.

        Args:
            x (torch.tensor): Input RGB image.
            latent_out (torch.tensor): Output from the last layer of an encoder.
            wb_gate (torch.tensor): White-balance gate used to change the influence of color scaled image.

        Returns:
            torch.tensor: returns White-Balanced image. 
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb,-log_wb_range,log_wb_range))
        
        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
        0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1)*wb
        wb_out = wb.unsqueeze(2).unsqueeze(3)*x
        wb_out = (wb_out-wb_out.min())/(wb_out.max()-wb_out.min())
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)*wb_out
        return wb_out
    
    def gamma_balance(self,x : torch.tensor,latent_out : torch.tensor,gamma_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            gamma_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma,-log_gamma,log_gamma))
        #print("gamma finally is ",gamma)
        g = torch.pow(torch.maximum(x,torch.tensor(1e-4)),gamma)
        g = (g-g.min())/(g.max()-g.min())
        g = g*gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g

    def sharpening(self,x : torch.tensor,latent_out: torch.tensor,sharpning_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            sharpning_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y,torch.tensor(0.1),torch.tensor(1.))
        s = x + (y*(x-out_x))
        s = (s-s.min())/(s.max()-s.min())
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s
    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            identity_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x

    def contrast(self,x : torch.tensor,latent_out : torch.tensor,contrast_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            contrast_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha) 
        contrast_image = (contrast_image-contrast_image.min())/(contrast_image.max()-contrast_image.min())
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image

    def tone(self,x : torch.tensor,latent_out : torch.tensor,tone_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            tone_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1,1,curve_steps)
        #print("tone_curve_ori=",tone_curve)
        #tone_curve=torch.tensor([[[1,1,1,1,0.1,0.1,0.1,0.1]]], device='cuda:0')
        tone_curve = self.tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i /curve_steps, 0, 1.0 /curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = (total_image-total_image.min())/(total_image.max()-total_image.min())
        total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image
    def bezier(self,x : torch.tensor,latent_out : torch.tensor,bezier_gate : torch.tensor):
        # Ensure the input image is clipped to the range [0, 1]
        img=x
        features=self.bezier_module(latent_out)
        #手动给值：
        #features=torch.tensor([[1,0,1,0,1,1,1,-1,1,0,1,0]])
        #features=torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0,1,0,1,0]])
        #print("输入features的shape是",features.shape)
        #features=features.repeat(1,3)
        batch_size = features.shape[0]
        params = features.view(batch_size, 3, 4)
        r1 = params[:, :, 0]  # [batch_size, 3]
        theta1 = params[:, :, 1]  # [batch_size, 3]
        r2 = params[:, :, 2]  # [batch_size, 3]
        theta2 = params[:, :, 3]  # [batch_size, 3]
        p1_x = (r1 + 1) / 2 * torch.cos((theta1 + 1) * torch.pi / 4)
        p1_y = (r1 + 1) / 2 * torch.sin((theta1 + 1) * torch.pi / 4)
        p2_x = 1 - (r2 + 1) / 2 * torch.cos((theta2 + 1) * torch.pi / 4)
        p2_y = 1 - (r2 + 1) / 2 * torch.sin((theta2 + 1) * torch.pi / 4)
        control_points = torch.stack([
          p1_x[:, 0], p2_x[:, 0], p1_y[:, 0], p2_y[:, 0],  # Red
          p1_x[:, 1], p2_x[:, 1], p1_y[:, 1], p2_y[:, 1],  # Green  
          p1_x[:, 2], p2_x[:, 2], p1_y[:, 2], p2_y[:, 2]   # Blue
        ], dim=1)

        #print("ToneFilter收到的feature是：",features,"（shape=",features.shape)
        #print("控制点为control_points",control_points,"shape是",control_points.shape)
        param = control_points.view(-1, 3, 4).unsqueeze(2).unsqueeze(2)
        param=param.cuda(0)
        img = torch.clamp(img, min=0.0, max=1.0)
        #print("基本的img的shape是",img.shape)
        #tone_curve_sum = param.sum(dim=4) + 1e-30  # Avoid division by zero
        total_image = torch.zeros_like(img)
        for channel in range(3):
          #print("channel=",channel)
          p1x=param[:,channel,0,0,0].unsqueeze(1)
          p2x=param[:,channel,0,0,1].unsqueeze(1)
          p1y=param[:,channel,0,0,2].unsqueeze(1)
          p2y=param[:,channel,0,0,3].unsqueeze(1)
          q = torch.linspace(0, 1, 8 + 1, device=param.device).unsqueeze(0)
          Cx = 3 * q * (1 - q)**2 * p1x + 3 * q**2 * (1 - q) * p2x + q**3
          Cy = 3 * q * (1 - q)**2 * p1y + 3 * q**2 * (1 - q) * p2y + q**3
          #print("Cx=",Cx,"Cy=",Cy)
          channel_img = img[:, channel:channel+1, :, :]
          #print("channel_img的shape是",channel_img.shape)
          #channel_param = param[:, channel:channel+1, :, :, :]
          #channel_curve_sum = channel_param.sum(dim=4) + 1e-30
          #print("channel_img=",channel_img)
          channel_output = torch.zeros_like(channel_img)
          for i in range(8):
              x_start = Cx[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [batch_size, 1, 1,1]
              y_start = Cy[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
              x_end = Cx[:, i+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)   # [batch_size, 1, 1,1]
              y_end = Cy[:, i+1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
              #print("x_start=",x_start,"x_end=",x_end,"shape是",x_start.shape)
              tone_step = torch.clamp(channel_img - 1.0 * x_start, min=torch.zeros_like(x_end), max=x_end-x_start)
              #print("tone_step的shape是",tone_step.shape)
              #total_image += tone_step * param[..., i]
              channel_output += tone_step * (y_end-y_start)/(x_end-x_start+ 1e-30)
              #print("param[...,i]的shape是",channel_output.shape)
          #print("channel_output=",channel_output)
          total_image[:, channel:channel+1, :, :] = channel_output
        total_image = total_image * bezier_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image
    def kernel(self, x,latent_out,kernel_gate):
        img=x
        paramraw=self.kernel_module(latent_out)

        #测试数据，手动调整kernel值：
        #paramraw=torch.tensor([[1,1,1,1,1,1]])
        paramraw = paramraw.cuda(0)
        batch_size = paramraw.shape[0]
        
        # 将特征重塑为两个卷积核的形状
        params = paramraw.view(batch_size, 2, 3, self.kernel_size, self.kernel_size)
        
        # 使用tanh将参数约束到[-1, 1]范围内
        param = torch.tanh(params)
        #print("para shape=",param.shape)
        img = torch.clamp(img, min=0.0, max=1.0)
        
        batch_size, channels, H, W = img.shape
        
        K1 = param[:, 0, :, :, :]  # (batch_size, 3, k, k)
        K2 = param[:, 1, :, :, :]  # (batch_size, 3, k, k)
        
        # 直接使用循环，确保每个元素都正确处理
        output = torch.zeros_like(img)
        
        for i in range(batch_size):
            for c in range(channels):
                channel_img = img[i:i+1, c:c+1, :, :]  # (1, 1, H, W)
                channel_K1 = K1[i:i+1, c:c+1, :, :]    # (1, 1, k, k)
                channel_K2 = K2[i:i+1, c:c+1, :, :]    # (1, 1, k, k)
                
                # padding
                padding = self.kernel_size // 2
                conv1 = torch.nn.functional.conv2d(channel_img, channel_K1, padding=padding)
                conv2 = torch.nn.functional.conv2d(channel_img, channel_K2, padding=padding)
                
                # 如果尺寸不匹配，裁剪到正确尺寸
                if conv1.shape[2] > H or conv1.shape[3] > W:
                    conv1 = conv1[:, :, :H, :W]
                if conv2.shape[2] > H or conv2.shape[3] > W:
                    conv2 = conv2[:, :, :H, :W]
                
                channel_output = channel_img * conv1 + conv2 + channel_img
                output[i:i+1, c:c+1, :, :] = channel_output
        
        output = torch.clamp(output, min=0.0, max=1.0)
        output = output * kernel_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return output
    def forward(self,x,linear_proj):
        gate = self.tanh_range(self.gate_module(linear_proj),0.01,1.0)

        #手动调整gate值：
        #因为现在gate还只有6维所以必须先手动给一个
        gate = torch.tensor([[0,0.8,0,0,0,0,0,0.1,0.2]],device='cuda:0')

        print("gate=",gate)
        #print("x shape是",x.shape,",linearprojshape是",linear_proj.shape)
        wb_out = self.white_balance(x,linear_proj,gate[:,0])
        gamma_out = self.gamma_balance(x,linear_proj,gate[:,1])
        identity_out = self.identity(x,gate[:,2])
        sharpening_out = self.sharpening(x,linear_proj,gate[:,3])
        fog_out = self.defog(x,linear_proj,gate[:,4])
        contrast_out = self.contrast(x,linear_proj,gate[:,5])
        tone_out = self.tone(x,linear_proj,gate[:,6])
        bezier_out=self.bezier(x,linear_proj,gate[:,7])
        kernel_out=self.kernel(x,linear_proj,gate[:,8])
        x = wb_out + gamma_out   + fog_out + sharpening_out + contrast_out + tone_out + identity_out+bezier_out+kernel_out
        #x=bezier_out+kernel_out
        x = (x-x.min())/(x.max()-x.min())
        return x,gate


class MultiLevelGDIP(torch.nn.Module):

    def __init__(self,
                encoder_output_dim : int = 256,
                num_of_gates : int = 7, reversed=False):

        super(MultiLevelGDIP,self).__init__()
        self.vision_encoder = VisionEncoder(encoder_output_dim,base_channel=32)
        self.gdip1 = GatedDIP(encoder_output_dim,num_of_gates)
        self.gdip2 = GatedDIP(encoder_output_dim,num_of_gates)
        self.gdip3 = GatedDIP(encoder_output_dim,num_of_gates)
        self.gdip4 = GatedDIP(encoder_output_dim,num_of_gates)
        self.gdip5 = GatedDIP(encoder_output_dim,num_of_gates)
        # self.gdip6 = GatedDIP(encoder_output_dim,num_of_gates)

        self.reversed = reversed
    
    def forward(self, x : torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # out_image = list()
        # gates_list = list()
        
        output_dict = self.vision_encoder(x)

        if not reversed:
            # Bottom-Up !

            x,gate_5 = self.gdip5(x,output_dict['linear_proj_5'])
            # out_image.append(x)
            # gates_list.append(gate_5)

            x,gate_4 = self.gdip4(x,output_dict['linear_proj_4'])
            # out_image.append(x)
            # gates_list.append(gate_4)

            x,gate_3 = self.gdip3(x,output_dict['linear_proj_3'])
            # out_image.append(x)
            # gates_list.append(gate_3)

            x,gate_2 = self.gdip2(x,output_dict['linear_proj_2'])
            # out_image.append(x)
            # gates_list.append(gate_2)

            x,gate_1 = self.gdip1(x,output_dict['linear_proj_1'])
            # out_image.append(x)
            # gates_list.append(gate_1)

            return x, gate_1

        else:
            # Top-Down !

            x,gate_1 = self.gdip1(x,output_dict['linear_proj_1'])
            # out_image.append(x)
            # gates_list.append(gate_1)
            
            x,gate_2 = self.gdip2(x,output_dict['linear_proj_2'])
            # out_image.append(x)
            # gates_list.append(gate_2)
            
            x,gate_3 = self.gdip3(x,output_dict['linear_proj_3'])
            # out_image.append(x)
            # gates_list.append(gate_3)
            
            x,gate_4 = self.gdip4(x,output_dict['linear_proj_4'])
            # out_image.append(x)
            # gates_list.append(gate_4)
            
            x,gate_5 = self.gdip5(x,output_dict['linear_proj_5'])
            # out_image.append(x)
            # gates_list.append(gate_5)

            return x, gate_5
        # return x,out_image,gates_list

if __name__ == '__main__':
    device = 'cpu'
    batch_size = 2
    encoder_out_dim = 256
    x = torch.randn(batch_size,3,448,448).to(device)
    x = (x-x.min())/(x.max()-x.min())
    model = MultiLevelGDIP(encoder_output_dim = encoder_out_dim).to(device)
    print(model)
    x,out_image,gates_list= model(x)
    print('x shape:',x.shape)
