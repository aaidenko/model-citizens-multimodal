"""Complete-image padding plus overlapping readable source tiles, without crops."""
import math
from PIL import Image, ImageOps


def padded(image,size=224):
    image=image.convert('RGB');small=ImageOps.contain(image,(size,size),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(size,size),'white')
    canvas.paste(small,((size-small.width)//2,(size-small.height)//2))
    return canvas


def positions(length,side):
    if length<=side:return [0]
    values=list(range(0,length-side+1,max(1,int(side*.75))))
    if values[-1]!=length-side:values.append(length-side)
    return values


def source_views(image,size=224,max_tiles=16):
    """First view always includes every source pixel; tiles cover the same image."""
    w,h=image.size;side=min(512,max(size,min(w,h)))
    while len(positions(w,side))*len(positions(h,side))>max_tiles:
        side=int(math.ceil(side*1.15))
    boxes=[(0,0,w,h)]
    for y in positions(h,side):
        for x in positions(w,side):
            box=(x,y,min(w,x+side),min(h,y+side))
            if box not in boxes:boxes.append(box)
    return [padded(image.crop(box),size) for box in boxes],boxes


def encoder_pixels(processor,views):
    # Padding/tiling has already constructed exact square inputs. Default CLIP
    # center-crop must not run again and silently discard source columns.
    return processor(images=views,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values']
