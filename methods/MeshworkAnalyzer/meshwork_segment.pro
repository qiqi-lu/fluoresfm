function meshwork_segment, image,subtractbackground=subtractbackground, minimumsize=minimumsize, threshold=threshold, interactive = interactive, groupleader = groupleader, default = defaultmask, stack= stack, $
  script = script, batch=batch
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common script, recordmode, scripttable, numberentry
  common segmentbatch, dobatch

  iscancel = 0
  recordmode = 0 & scripttable = strarr(4,25) & numberentry = 0
  stackmode = 0

  if keyword_set(script) then doscript = 1 else doscript = 0
  if size(image,/n_dimensions) lt 2 then return, -1
  if keyword_set(interactive) and keyword_set(stack) then begin
    if size(stackproc, /type) eq 0 then return, -1
    if size(famaskbinary, /type) eq 0 then return, -1

    faimage = stackproc[*,*,0]
    grayimage = stackproc[*,*,0]
    oldwindow = !d.window
    stackmode = stack
    case stackmode of
      1: binaryimage = famaskbinary[*,*,0]
      2: binaryimage = cellmaskdynamic[*,*,0]
    endcase
    WID_BASE_sfa, GROUP_LEADER=groupleader, _EXTRA=_VWBExtra_,stack=stack,script=doscript
    return, binaryimage
  endif

  width = (size(image,/dimensions))[0]
  height = (size(image,/dimensions))[1]
  faimage = image
  grayimage = image

  if keyword_set(subtractbackground) and ~keyword_set(stack) then grayimage = SternbergBackgroundSubtraction(image,background=background, radius = 10,scaling = 0.15,/interactive,groupleader=groupleader) $
  else grayimage = image

  if keyword_set(defaultmask) then binaryimage =  defaultmask else begin
    binaryimage = faimage*0
    defaultmask = binaryimage
    ;print, 'here'
  endelse

  if keyword_set(interactive) and ~keyword_set(stack) then begin
    oldwindow = !d.window
    WID_BASE_sfa, GROUP_LEADER=groupleader, _EXTRA=_VWBExtra_,script=doscript
    if iscancel then return, defaultmask else return, binaryimage
  endif else begin

    minsize = 0 & if keyword_set(minimumsize) and ~keyword_set(stack) then minsize = minimumsize
    thres = 100 & if keyword_set(threshold) and ~keyword_set(interactive) then thres = threshold else if ~keyword_set(interactive) then return, -1

    mask = image*0+1
    belowthres = where(grayimage ge thres)
    mask[belowthres] = 0
    grayimage[belowthres] = 0

    if keyword_set(minimumsize) and ~keyword_set(stack) then begin
      mask = removesmallpatch(mask,minsize)
      grayimage = grayimage*mask
    endif

  endelse


  return, binaryimage
end


pro appendscript, event, script=scriptstruct
  common script, recordmode, scripttable, numberentry

  if size(scriptstruct, /type) ne 8 then return
  if numberentry gt 24 then return

  ;print, 'here'
  scripttable[0,numberentry] = string(scriptstruct.actioncode)
  scripttable[1,numberentry] = string(scriptstruct.parameter1)
  scripttable[2,numberentry] = string(scriptstruct.parameter2)
  scripttable[3,numberentry] = string(scriptstruct.parameter3)

  numberentry = numberentry+1
  updatescripttable, event

end

pro togglescript, event, stop=stop
  common script, recordmode, scripttable, numberentry

  if keyword_set(stop) then begin
    recordmode = 0
    widget_control, widget_info(event.top, find_by_uname='WID_BUTTON_SCRIPTSTATUS'),set_value='Not recording'
  endif

  recordmode = recordmode? 0:1
  if recordmode eq 1 then widget_control, widget_info(event.top, find_by_uname='WID_BUTTON_SCRIPTSTATUS'),set_value='Recording' else $
    widget_control, widget_info(event.top, find_by_uname='WID_BUTTON_SCRIPTSTATUS'),set_value='Not recording'
end

pro executescript, event, cellmask = cellmask, quick = quick, batch=batch, single=single
  common script, recordmode, scripttable, numberentry
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode = stack
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic

  if keyword_set(batch) then begin
    if keyword_set(single) then begin
      for i = 0, numberentry-1 do begin
        action =  getscriptentry(event, i)
        runscript, action
        if keyword_set(cellmask) then cellmaskbinary = binaryimage
      endfor
      return
    endif else begin
      for j = 0, properties.frames-1 do begin
        ;if ~keyword_set(batch) then widget_control,widget_info(event.top, find_by_uname='WID_SLIDER_FRAME'),set_value=j
        ;if ~keyword_set(batch) then showmaskoverlay, event, frame=j
        faimage = stackproc[*,*,j]
        grayimage = stackproc[*,*,j]
        ;binaryimage = grayimage*0
        if keyword_set(cellmask) then binaryimage = cellmaskdynamic[*,*,j] else binaryimage = famaskbinary[*,*,j]
        ;
        print, 'Executing frame:', j
        for i = 0, numberentry-1 do begin
          action =  getscriptentry(event, i)
          runscript, action
          ; if ~keyword_set(batch) then showfaimage, event
          ; if ~keyword_set(batch) then showbinaryimage, event
          ;plotboundary,oldbin, color='green',/boundary
          ;  if ~keyword_set(quick) and ~keyword_set(batch) then plotboundary, binaryimage, color='red',/boundary
          ;wait,0.5
        endfor

        if ~keyword_set(cellmask) then famaskbinary[*,*,j] = binaryimage else cellmaskdynamic[*,*,j] = binaryimage
        ; if ~keyword_set(quick) and ~keyword_set(batch)then showfaimage, event
        ; if ~keyword_set(quick) and ~keyword_set(batch) then plotboundary, binaryimage, color='green',/boundary,thick = 1.5
      endfor
      return
    endelse
  endif


  if keyword_set(single) then begin
    for i = 0, numberentry-1 do begin
      action =  getscriptentry(event, i)
      runscript, action
      showbinaryimage, event
    endfor
    showbinaryimage, event
    return
  endif

  if size(properties, /type) ne 8 then return
  if size(stackproc, /type) eq 0 then return
  if size(cellmaskdynamic, /type) eq 0 then return
  if size(famaskbinary, /type) eq 0 then return

  print, 'Number of script entry: ', numberentry
  print, 'Number of frames: ',properties.frames

  for j = 0, properties.frames-1 do begin
    if ~keyword_set(batch) then widget_control,widget_info(event.top, find_by_uname='WID_SLIDER_FRAME'),set_value=j
    if ~keyword_set(batch) then showmaskoverlay, event, frame=j
    faimage = stackproc[*,*,j]
    grayimage = stackproc[*,*,j]
    ;binaryimage = grayimage*0
    if keyword_set(cellmask) then binaryimage = cellmaskdynamic[*,*,j] else binaryimage = famaskbinary[*,*,j]
    ;
    print, 'Executing frame:', j
    for i = 0, numberentry-1 do begin
      action =  getscriptentry(event, i)
      runscript, action
      if ~keyword_set(batch) then showfaimage, event
      if ~keyword_set(batch) then showbinaryimage, event
      ;plotboundary,oldbin, color='green',/boundary
      if ~keyword_set(quick) and ~keyword_set(batch) then plotboundary, binaryimage, color='red',/boundary
      ;wait,0.5
    endfor

    if ~keyword_set(cellmask) then famaskbinary[*,*,j] = binaryimage else cellmaskdynamic[*,*,j] = binaryimage
    if ~keyword_set(quick) and ~keyword_set(batch)then showfaimage, event
    if ~keyword_set(quick) and ~keyword_set(batch) then plotboundary, binaryimage, color='green',/boundary,thick = 1.5
  endfor

  print,'Script execution completed'
  if keyword_set(quick) and ~keyword_set(batch) then showfaimage, event
  if keyword_set(quick) and ~keyword_set(batch) then plotboundary, binaryimage, color='green',/boundary,thick = 1.5
end


pro testscript, event
  common script, recordmode, scripttable, numberentry
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode = stack
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic

  print, 'Number of entry: ', numberentry

  for i = 0, numberentry-1 do begin
    action =  getscriptentry(event, i)
    runscript, action
    showfaimage, event
    showbinaryimage, event
    ;plotboundary,oldbin, color='green',/boundary
    ;plotboundary, binaryimage, color='red',/boundary
    wait,0.5
  endfor

  showfaimage, event
  plotboundary, binaryimage, color='salmon',/boundary
end

pro updatescripttable, event, clear = clear
  common script, recordmode, scripttable, numberentry

  if keyword_set(clear) then begin
    widget_control, widget_info(event.top,find_by_uname='WID_TABLE_SCRIPT'),set_value =strarr(4,25)
    return
  endif

  if size(scripttable, /type) eq 0 then return
  if numberentry eq 0 then return
  widget_control, widget_info(event.top,find_by_uname='WID_TABLE_SCRIPT'),set_value =scripttable

end

function getscriptentry, event, number
  common script, recordmode, scripttable, numberentry
  if number gt numberentry-1 then return, -1
  return, {actioncode:uint(scripttable[0,number]), parameter1:uint(scripttable[1,number]),parameter2:uint(scripttable[2,number]),parameter3:uint(scripttable[3,number])}

end

pro clearscript, event
  common script, recordmode, scripttable, numberentry
  scripttable = strarr(4,25)
  numberentry = 0
  updatescripttable, event,/clear
end

pro runscript, scriptstruct
  common script, recordmode, scripttable, numberentry
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode = stack

  if size(scriptstruct,/type) ne 8 then return
  case scriptstruct.actioncode of
    -1: grayimage = double(faimage)
    1: case scriptstruct.parameter1 of
    0:grayimage = median(double(grayimage), 3)
    1:grayimage = gauss_smooth(double(grayimage), /edge_truncate)
  endcase
  2:     grayimage = SternbergBackgroundSubtraction(grayimage, radius = scriptstruct.parameter1,scaling = scriptstruct.parameter2)
  3: case scriptstruct.parameter1 of
  1: binarize, event, direct=1,update= scriptstruct.parameter3
  2: binarize, event, direct=2,update= scriptstruct.parameter3
  3: binarize, event, direct=3,update= scriptstruct.parameter3
  4: binarize, event, direct=4,update= scriptstruct.parameter3
  5: binarize, event, direct=5,update= scriptstruct.parameter3
  6: binarize, event, direct=6,update= scriptstruct.parameter3
  7: binarize, event, direct=7,update= scriptstruct.parameter3
  8: binarize, event, direct=8,update= scriptstruct.parameter3
  9: binarize, event, direct=9, thres = scriptstruct.parameter2,update= scriptstruct.parameter3
  10:binarize, event, direct=10, thres = scriptstruct.parameter2,update= scriptstruct.parameter3
endcase
4: binarize, event, direct =-1, thres=scriptstruct.parameter2
5: morphological, event, direct= scriptstruct.parameter1
6: case scriptstruct.parameter1 of
1: filterbysize, event,/small, min=scriptstruct.parameter2
2: filterbysize, event,/large, max=scriptstruct.parameter2
3: filterbysize, event,/fill, min=scriptstruct.parameter2
endcase
endcase

end

pro savescript, event
  common script, recordmode, scripttable, numberentry

  fn = Dialog_Pickfile(/write,get_path=fpath,file=ref_file, filter=['*script.sav'],title='Save to *script.sav file')
  cd,fpath
  if fn eq '' then return

  mapfile=AddExtension(fn,'_script.sav')

  save,scripttable, numberentry, $
    filename = mapfile

  print, 'save finished'
  if ~keyword_set(ff) then result = dialog_message('Saving completed')

end

pro loadscript, event, batch=batch, filename= file, loadflag= loadflag
  common script, recordmode, scripttable, numberentry
  loadflag = 1
  if ~keyword_set(file) then begin
    filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*script.sav'],title='Select *script.sav file to open')
    if filename eq '' then begin
      print,'filename not recognized', filename
      if keyword_set(loadflag) then loadflag = -1
      return
    endif
    cd,fpath
  endif else filename = file

  print,'opening file: ', filename
  if strpos(filename,'script.sav') ne -1 then restore,filename=filename else begin
    print, 'Incorrect filetype'
    if keyword_set(loadflag) then loadflag = -1
    return
  end
  if ~keyword_set(batch) then updatescripttable, event

end

pro showfaimage, event, bw=bw, frame= frame, widget=widget

  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters

  wxsz = 1024 & wysz = 1024

  if ~keyword_set(widget) then widget=event.top

  if keyword_set(frame) and stackmode ne 0 then begin
    if size(properties, /type) eq 0 then return
    if size(stackraw, /type) eq 0 then return
    if size(stackproc, /type) eq 0 then return
    colormode = widget_info(widget_info(widget,find_by_uname='WID_DROPLIST_COLOR'),/droplist_select)
    ctx = [0,13,5,25,3,33,34,0]
    ct = ctx[colormode]
    if colormode eq 7 then cgloadct, ct, /reverse

    xmin = 0 & xmax = properties.xpixels-1
    ymin = 0 & ymax = properties.ypixels-1
    im = stackproc[xmin:xmax,ymin:ymax,frame]
    imscl = congrid(im, wxsz, wysz)
    ;device, decompose = 0
    cgimage, bytscl(imscl)
    return
  endif

  if size(faimage, /type) eq 0 then return
  if size(faimage,/n_dimensions) ge 2 then  im = faimage[*,*,0] else return
  imscl = congrid(im, wxsz, wysz,/interp)

  colormode = widget_info(widget_info(widget,find_by_uname='WID_DROPLIST_COLOR'),/droplist_select)
  ctx = [0,13,5,25,3,33,34,0]
  ct = ctx[colormode]
  cgloadct, ct
  if colormode eq 7 then cgloadct, ct, /reverse

  if keyword_set(bw) then loadct,0

  ;device, decompose = 0
  cgimage, bytscl(imscl)
end

pro showgrayimage, event, frame= frame, widget=widget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters

  if ~keyword_set(widget) then widget=event.top
  wxsz = 1024 & wysz = 1024

  if keyword_set(frame) and stackmode ne 0 then begin
    if size(properties, /type) eq 0 then return
    if size(stackraw, /type) eq 0 then return
    if size(stackproc, /type) eq 0 then return
    colormode = widget_info(widget_info(widget,find_by_uname='WID_DROPLIST_COLOR'),/droplist_select)
    ctx = [0,13,5,25,3,33,34,0]
    ct = ctx[colormode]
    if colormode eq 7 then cgloadct, ct, /reverse

    xmin = 0 & xmax = properties.xpixels-1
    ymin = 0 & ymax = properties.ypixels-1
    im = stackproc[xmin:xmax,ymin:ymax,frame]
    imscl = congrid(im, wxsz, wysz)
    device, decompose = 0
    tvscl, imscl
    return
  endif

  if size(grayimage,/type) eq 0 then return

  if size(grayimage,/n_dimensions) ge 2 then  im = grayimage[*,*,0] else return
  imscl = congrid(im, wxsz, wysz)

  colormode = widget_info(widget_info(widget,find_by_uname='WID_DROPLIST_COLOR'),/droplist_select)
  ctx = [0,13,5,25,3,33,34,0]
  ct = ctx[colormode]
  if colormode eq 7 then cgloadct, ct, /reverse
  ;loadct, 3
  device, decompose = 0
  tvscl, imscl
end

pro showbinaryimage, event, frame=frame, overlayframe=overlayframe, widget=widget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common famask, famaskbinary

  wxsz = 1024 & wysz = 1024

  if keyword_set(frame) and stackmode ne 0 then begin
    if size(famaskbinary, /type) eq 0 then return
    im = binaryimage[*,*]
    imscl = congrid(im, wxsz, wysz)
    if keyword_set(overlayframe) then cgImage, imscl,ctIndex = 0, transparent = 70,position = [0,0,1,1] else $
      tvscl, imscl
    return
  endif

  if size(binaryimage,/type) eq 0 then return

  if size(binaryimage,/n_dimensions) ge 2 then  im = binaryimage[*,*,0] else return
  ; imscl = congrid(im, wxsz, wysz)
  ;loadct, 3
  ;device, decompose = 0
  tvscl, congrid(im, wxsz, wysz)
end

pro showmaskoverlay, event, bw=bw, frame= frame, widget=widget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters

  wxsz = 1024 & wysz = 1024

  if size(binaryimage,/type) eq 0 then return

  if keyword_set(frame) and stackmode ne 0 then begin
    cgimage, bytarr(wxsz,wysz)
    showfaimage, event, frame=frame, widget=widget
    showbinaryimage, event, frame=frame,/overlayframe, widget=widget
    return
  endif

  scaledmask =  congrid(bytscl(~binaryimage, min =0, max = 1),wxsz, wysz)
  ;showfaimage, event
  cgimage, bytarr(wxsz,wysz)
  showfaimage, event, widget=widget
  xt = keyword_set(bw) ? 0:13
  cgImage, scaledmask,ctIndex = xt, transparent = 70,position = [0,0,1,1]

end


pro filterbysize, event, small=small, large = large, fill=fill, script= scriptstruct, min=min, max=max
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  if ~keyword_set(min) and ~keyword_set(max) then begin
    widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_MIN'),get_value = minsize
    widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_MAX'),get_value = maxsize
  endif else begin
    if keyword_set(min) then minsize = min
    if keyword_set(max) then maxsize = max
  endelse


  if keyword_set(fill) then begin
    binaryimage = fillsmallholes(binaryimage, minsize)
    scriptstruct = {actioncode: 6, parameter1: 3 , parameter2: minsize, parameter3:0 }
    return
  endif
  if keyword_set(small) and ~keyword_set(large) then begin
    binaryimage = removesmallpatch(binaryimage,minsize)
    scriptstruct = {actioncode: 6, parameter1: 1 , parameter2: minsize, parameter3:0 }
    return
  endif
  if ~keyword_set(small) and keyword_set(large) then begin
    binaryimage = removelargepatch(binaryimage,maxsize)
    scriptstruct = {actioncode: 6, parameter1: 2 , parameter2: maxsize, parameter3:0 }
    return
  endif
  binaryimage = removelargepatch(removesmallpatch(binaryimage,minsize),maxsize)
  scriptstruct = replicate({actioncode: 6, parameter1: 1 , parameter2: minsize, parameter3:0 },2)
  scriptstruct[1].actioncode = 6
  scriptstruct[1].parameter1 = 2
  scriptstruct[1].parameter2 =maxsize
end

function morph_boundary, binimage
  strel = replicate(1,3,3)
  er = erode(binimage,strel)
  return, binimage and ~er
end

function morph_holefilling, binimage
  strel = replicate(1,3,3)
  dil = dilate(binimage,strel)
  return, binimage
end


pro morphological, event, script=scriptstruct, direct=direct
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow
  ;
  ;'Dilate','Erode','Close','Open','Thin','Thicken','Fill'
  if ~keyword_set(direct) then mmode = 1+widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_MORPH'),/droplist_select) else mmode=direct
  ;strel = replicate(1,3,3)
  ;strel[0,0] = 0
  ;strel[0,2] = 0
  ;strel[2,0] = 0
  ;strel[2,2] = 0
  strel = [[0,1,0],[1,1,1],[0,1,0]]
  case mmode of
    1:begin ; dilate
      binaryimage = dilate(binaryimage, strel)
      scriptstruct = {actioncode: 5, parameter1:1, parameter2: 0 , parameter3: 0}
    end
    2: begin ; erode
      binaryimage = erode(binaryimage, strel)
      scriptstruct = {actioncode: 5, parameter1:2, parameter2: 0, parameter3: 0}
    end
    3:begin ; close
      binaryimage = morph_close(binaryimage, strel)
      scriptstruct = {actioncode: 5, parameter1:3, parameter2: 0, parameter3: 0}
    end
    4:begin ; open
      binaryimage = morph_open(binaryimage, strel)
      scriptstruct = {actioncode: 5, parameter1:4, parameter2: 0, parameter3: 0}
    end
    5:begin ; thin
      binaryimage = morph_thin(binaryimage,strel,strel)
      scriptstruct = {actioncode: 5, parameter1:5, parameter2: 0, parameter3: 0}
    end
  endcase
end

pro binarize, event, script=scriptstruct, direct=directoption, thres=thres, update=update
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode

  if keyword_set(directoption) then begin
    if keyword_set(thres) then threshold = thres
    tmode = directoption+1
  endif else begin
    widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_THRES'),get_value = threshold
    tmode = 1+widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_THRESHOLD'),/droplist_select)
    update = widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_UPDATE'),/droplist_select)
    widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_MIN'),get_value = minsize

    ;['Simple Threshold (Slider)','Otsu','Mean','Isodata','Moments','Max Entropy','Min Error']
  endelse

  case tmode of
    1: begin ; simple
      binaryimage = combinebinary(binaryimage,grayimage ge threshold,update=update)
      scriptstruct = {actioncode: 4, parameter1:threshold, parameter2: 0 , parameter3: update}
    end
    2: begin ; 'otsu'
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/otsu,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:1, parameter2: 0 , parameter3: update}
    end
    3: begin ; 'mean'
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/mean,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:2, parameter2: 0 , parameter3: update}
    end
    4: begin ; isodata
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/isodata,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:3, parameter2: 0 , parameter3: update}
    end
    5: begin ; moments
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/moments,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:4, parameter2: 0 , parameter3: update}
    end
    6: begin ;max entropy
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/maxenetropy,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:5, parameter2: 0 , parameter3: update}
    end
    7: begin ; minerror
      binaryimage = combinebinary(binaryimage,image_threshold(grayimage,/minerror,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:6, parameter2: 0 , parameter3: update}
    end
    8: begin ; Rosin-Otsu
      binaryimage = combinebinary(binaryimage,threshold_otsu_rosin(grayimage,threshold = ts),update=update)
      scriptstruct = {actioncode: 3, parameter1:7, parameter2: 0 , parameter3: update}
    end
    9: begin ; Rosin-Otsu
      binaryimage = combinebinary(binaryimage,threshold_otsu_rosin(grayimage,threshold = ts,ratio=[2,1]),update=update)
      scriptstruct = {actioncode: 3, parameter1:8, parameter2: 0 , parameter3: update}
    end
    10: begin ; variable Rosin-Otsu
      tx = ((threshold <100)>0) /100.
      ratio = [tx, 1-tx]
      binaryimage = combinebinary(binaryimage,threshold_otsu_rosin(grayimage,threshold = ts,ratio=ratio),update=update)
      scriptstruct = {actioncode: 3, parameter1:9, parameter2: threshold , parameter3: update}
    end
    11: begin ; simple threshold on mask data
      binaryimage = combinebinary(binaryimage,(grayimage*binaryimage) ge threshold,update=update)
      scriptstruct = {actioncode: 10, parameter1:threshold, parameter2: 0 , parameter3: update}
    end
    ;  12: begin; Zamir-Soferman
    ;      binaryimage = soferman(image=grayimage,minpatchsize = minsize ,threshold = threshold)
    ;      cgimage, binaryimage,position=[0,0,1,1],ctindex =13
    ;      scriptstruct = {actioncode: 11, parameter1:minsize, parameter2: threshold}
    ;   end
  endcase
end

pro quit_WID_BASE_SFA, wWidget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common script, recordmode, scripttable, numberentry
  common interactive, imode, drawwindow, undoimage
  common display, wxsz, wysz, mainwindow, zoomcoord, autoscale, screenmode

  ;iscancel = 1
  wset, mainwindow

end


pro WID_BASE_sfa_event, Event
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common script, recordmode, scripttable, numberentry
  common interactive, imode, drawwindow, undoimage

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
    widget_info(Event.id, /tree_root) : event.id)
  wWidget =  Event.top

  if stackmode ne 0 then widget_control,widget_info(wwidget,find_by_uname = 'WID_SLIDER_FRAME'),get_value = thisframe else thisframe = -1

  case wTarget of
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWORIG'): showfaimage, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWGRAY'): showgrayimage, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DRAW_MAIN'): sfadrawevents, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RESET'): begin
      case stackmode of
        0: begin
          grayimage = faimage
          showfaimage, event
        end
        1:begin
          if size(stackproc, /type) eq 0 then return
          if size(famaskbinary, /type) eq 0 then return
          faimage = stackproc[*,*,thisframe]
          grayimage = stackproc[*,*,thisframe]
          binaryimage = famaskbinary[*,*,thisframe]
        end
        2: begin
          if size(stackproc, /type) eq 0 then return
          if size(cellmaskdynamic, /type) eq 0 then return
          faimage = stackproc[*,*,thisframe]
          grayimage = stackproc[*,*,thisframe]
          binaryimage = cellmaskdynamic[*,*,thisframe]
        end
      endcase
      if recordmode eq 1 then appendscript, event, script= {actioncode:-1,parameter1:0,parameter2:0,parameter3:0}
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_BACKSUB'): begin
      if recordmode eq 0 then grayimage = SternbergBackgroundSubtraction(grayimage,background=background, radius = 10,scaling = 0.15,/interactive,groupleader=event.top) else begin
        grayimage = SternbergBackgroundSubtraction(grayimage,background=background, radius = 10,scaling = 0.15,/interactive,groupleader=event.top, script=scriptstruct)
        if size(scriptstruct,/type) eq 8 then appendscript, event, script=scriptstruct
      endelse
      showgrayimage, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_THRES'): begin
      if recordmode eq 1 then binarize, event, script=scriptstruct else binarize, event
      showbinaryimage,event
      if size(scriptstruct,/type) eq 8 then appendscript, event, script=scriptstruct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWMASK'): begin
      showbinaryimage, event
      if size(scriptstruct,/type) eq 8 then appendscript, event, script=scriptstruct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWMASKOVERLAY'):         showmaskoverlay, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_MORPH'): begin
      if recordmode eq 1 then morphological, event, script=scriptstruct else morphological, event
      showbinaryimage, event
      if size(scriptstruct,/type) eq 8 then appendscript, event, script=scriptstruct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SIZE'): begin
      if recordmode eq 1 then filterbysize, event, script=scriptstruct else filterbysize, event
      showbinaryimage, event
      help, scriptstruct
      if size(scriptstruct, /type) eq 8 then begin
        if n_elements(scriptstruct) eq 1 then appendscript, event, script=scriptstruct else begin
          appendscript, event, script=scriptstruct[0]
          appendscript, event, script=scriptstruct[1]
        endelse
      endif
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SIZES'): begin
      if recordmode eq 1 then filterbysize, event,/small, script=scriptstruct else filterbysize, event,/small
      showbinaryimage, event
      if size(scriptstruct, /type) eq 8 then appendscript, event, script=scriptstruct
      ;help, scriptstruct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_DONE'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_BUTTON' )then begin
        iscancel = 0
        widget_control, event.top,/destroy
        wset, oldwindow
      endif
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CANCEL'): begin
      iscancel = 1
      widget_control, event.top,/destroy
      wset, oldwindow
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_MSMOOTH'): begin
      grayimage = median(grayimage, 3)
      showgrayimage, event
      if recordmode eq 1 then appendscript, event, script={actioncode:1,parameter1:0,parameter2:0,parameter3:0}
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GSMOOTH'): begin
      grayimage = gauss_smooth(double(grayimage), /edge_truncate)
      showgrayimage, event
      if recordmode eq 1 then appendscript, event, script={actioncode:1,parameter1:1,parameter2:0,parameter3:0}
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_THRES'): begin
      binarize, event
      showbinaryimage,event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_COLOR'):  showfaimage, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_FILL'): begin
      if recordmode eq 1 then filterbysize, event, /fill, script=scriptstruct else filterbysize, event, /fill
      showbinaryimage, event
      if size(scriptstruct, /type) eq 8 then appendscript, event, script=scriptstruct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_FRAME'): begin
      case stackmode of
        0:
        1: begin
          if size(stackproc, /type) eq 0 then return
          faimage = stackproc[*,*,event.value]
          grayimage = stackproc[*,*,event.value]
          if size(famaskbinary, /type) eq 0 then return
          binaryimage = famaskbinary[*,*,event.value]
          showmaskoverlay, event
        end
        2: begin
          if size(stackproc, /type) eq 0 then return
          faimage = stackproc[*,*,event.value]
          grayimage = stackproc[*,*,event.value]
          if size(cellmaskdynamic, /type) eq 0 then return
          binaryimage = cellmaskdynamic[*,*,event.value]
          showmaskoverlay, event
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMMITPROCEED'): begin
      if size(famaskbinary,/type) eq 0 then return
      if size(binaryimage, /type) eq 0 then return
      case stackmode of
        0: begin
        end
        1: begin
          if size(stackproc, /type) eq 0 then return
          if size(famaskbinary, /type) eq 0 then return
          famaskbinary[*,*,thisframe] = binaryimage
          thisframe = thisframe+1 <(properties.frames-1)
          widget_control,widget_info(wwidget, find_by_uname='WID_SLIDER_FRAME'),set_value=thisframe
          faimage = stackproc[*,*,thisframe]
          grayimage = stackproc[*,*,thisframe]
          binaryimage = famaskbinary[*,*,thisframe]
          showmaskoverlay, event
        end
        2: begin
          if size(cellmaskdynamic, /type) eq 0 then return
          if size(stackproc, /type) eq 0 then return
          cellmaskdynamic[*,*,thisframe] = binaryimage
          thisframe = thisframe+1 <(properties.frames-1)
          widget_control,widget_info(wwidget, find_by_uname='WID_SLIDER_FRAME'),set_value=thisframe
          faimage = stackproc[*,*,thisframe]
          grayimage = stackproc[*,*,thisframe]
          binaryimage = cellmaskdynamic[*,*,thisframe]
          showmaskoverlay, event
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMMIT'): begin
      if size(famaskbinary,/type) eq 0 then return
      if size(binaryimage, /type) eq 0 then return
      case stackmode of
        0: begin
        end
        1: begin
          if size(stackproc, /type) eq 0 then return
          if size(famaskbinary, /type) eq 0 then return
          famaskbinary[*,*,thisframe] = binaryimage
        end
        2: begin
          if size(cellmaskdynamic, /type) eq 0 then return
          if size(stackproc, /type) eq 0 then return
          cellmaskdynamic[*,*,thisframe] = binaryimage
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMPAREFAMASK'): begin
      case stackmode of
        0: begin
        end
        1: begin
          if size(famaskbinary,/type) eq 0 then return
          if size(binaryimage, /type) eq 0 then return
          showfaimage, event
          oldbin = famaskbinary[*,*,thisframe]
          plotboundary,oldbin, color='green',/boundary
          plotboundary, binaryimage, color='red',/boundary
        end
        2: begin
          if size(cellmaskdynamic,/type) eq 0 then return
          if size(binaryimage, /type) eq 0 then return
          showfaimage, event
          oldbin = cellmaskdynamic[*,*,thisframe]
          plotboundary,oldbin, color='green',/boundary
          plotboundary, binaryimage, color='red',/boundary

        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TOGGLESCRIPT'):togglescript, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CLEARSCRIPT'):clearscript, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TESTSCRIPT'):testscript, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SAVESCRIPT'):savescript, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LOADSCRIPT'):loadscript, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXECUTESCRIPT'):begin
      case stackmode of
        1: executescript, event
        2: executescript, event,/cellmask
        else: executescript, event,/single
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXECUTESCRIPTDARK'):begin
      case stackmode of
        1: executescript, event,/quick
        2: executescript, event,/cellmask,/quick
        else:  executescript, event,/single
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_MOUSE'):imode= widget_info(widget_info(wwidget,FIND_BY_UNAME='WID_DROPLIST_MOUSE'),/droplist_select)
    ;     Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_HISTOGRAMINTENSITY'): begin
    ;       case widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_HISTO'),/droplist_select) of
    ;         0: hh = grayimage
    ;         1: hh = grayimage*binaryimage
    ;         2: hh = grayimage*cellmaskbinary
    ;       endcase
    ;       if (min(hh)-max(hh)) eq 0 then return else cghistoplot, hh ,mininput= 1, maxinput= 65534
    ;     end
    else:
  endcase
end

pro initializesfa, wWidget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common mouse, mousedown, infoinitial, mousemode
  common interactive, imode, drawwindow, undoimage

  wxsz = 1024 & wysz = 1024
  imode = 0
  case stackmode of
    0: begin
      if size(binaryimage,/type )eq 0 then binaryimage = faimage*0
      grayimage = faimage
    end
    1: begin
      if size(famaskbinary, /type) ne 0 then binaryimage = famaskbinary[*,*,0] else binaryimage = faimage*0
      if size(stackproc, /type) eq 0 then grayimage= faimage else grayimage = stackproc[*,*,0]
    end
    2: begin
      if size(cellmaskdynamic, /type) ne 0 then binaryimage = cellmaskdynamic[*,*,0] else binaryimage = faimage*0
      if size(stackproc, /type) eq 0 then grayimage= faimage else grayimage = stackproc[*,*,0]

    end
  endcase

  mainwindow= !D.WINDOW
  ;print, 'mainwindow',mainwindow
  boxcolor = !D.N_colors-10
  info = {image: bytarr(wxsz,wysz), $
    wid:mainwindow, $
    drawID: widget_info(wWidget,find_by_uname='WID_DRAW_MAIN'), $
    pixID:-1, $
    xsize:wxsz, $
    ysize:wysz, $
    sx:-1,$
    sy:-1,$
    boxColor: boxColor}
  infoinitial = info
  widget_control, wWidget, set_uvalue =info, /no_copy


  ;
  ;
  ;
  ;  if size(faimage,/n_dimensions) ge 2 then  im = faimage[*,*,0] else return
  ;  imscl = congrid(im, wxsz, wysz,/interp)
  showmaskoverlay, event, widget=wwidget

end

pro WID_BASE_sfa, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_, stack=stack, script=script
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common script, recordmode, scripttable, numberentry

  if keyword_set(stack) then begin
    case stack of
      0: tt = 'Region Segmentation (c) 2013-2018 Kanchanawong Lab, MBI/NUS www.nanoscalemechanobiology.org'
      1: tt = 'Focal Adhesion Segmentation (c) 2013-2018 Kanchanawong Lab, MBI/NUS  www.nanoscalemechanobiology.org'
      2: tt = 'Cell Segmentation (c) 2013-2018 Kanchanawong Lab, MBI/NUS  www.nanoscalemechanobiology.org'
    endcase
  endif else tt =   'Region Segmentation (c) 2013-2018 Kanchanawong Lab, MBI/NUS  www.nanoscalemechanobiology.org'

  if keyword_set(script) then xpanelsize = 1650 else xpanelsize =1300

  WID_BASE_sfa = Widget_Base( GROUP_LEADER=wGroup,  $
    UNAME='wid_base_sfa' , XOFFSET=5 ,YOFFSET=5  $
    ,SCR_XSIZE=xpanelsize ,SCR_YSIZE=1060,XSIZE=xpanelsize,YSIZE=1060,/modal,notify_realize='initializesfa' $
    , TITLE=tt ,SPACE=3 ,XPAD=3 ,YPAD=3,kill_notify='quit_WID_BASE_SFA' )

  WID_DRAW_main = Widget_Draw(WID_BASE_sfa,  $
    UNAME='WID_DRAW_MAIN' ,XOFFSET=210+50 ,YOFFSET=5  $
    ,SCR_XSIZE=1024 ,SCR_YSIZE=1024  $
    ,/BUTTON_EVENTS)

  if keyword_set(stack) then WID_slider_frame =   Widget_Slider(WID_BASE_sfa,  $
    UNAME='WID_SLIDER_FRAME' ,XOFFSET=215 ,YOFFSET=5  $
    ,SCR_XSIZE=45 ,SCR_YSIZE=1000 ,TITLE='',MAXIMUM=properties.frames-1,value = 0.,/vertical)

  if keyword_set(script) then begin
    WID_BUTTON_recordscript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_TOGGLESCRIPT' ,XOFFSET=1024+10+260 ,YOFFSET=10  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Toggle Script recording')
    WID_BUTTON_script = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_SCRIPTSTATUS' ,XOFFSET=1024+10+260+130 ,YOFFSET=10  $
      ,SCR_XSIZE=80 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Not recording')
    WID_BUTTON_clearscript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_CLEARSCRIPT' ,XOFFSET=1024+10+260+110*2 ,YOFFSET=10  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Clear script')
    WID_Table_batcher = Widget_Table(wid_base_sfa,  $
      UNAME='WID_TABLE_SCRIPT' ,xoffset = 1024+10+260, YOFFSET=10+30+10 ,SCR_XSIZE=350  $
      ,SCR_YSIZE=430 ,COLUMN_LABELS=[ 'Action', 'Param 1',  $
      'Param 2', 'Param 3'] ,$
      XSIZE=4 ,YSIZE=25,column_widths=[100,80,80,80])
    WID_BUTTON_TEstscript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_TESTSCRIPT' ,XOFFSET=1024+10+260+110*0 ,YOFFSET=10+40+430+10  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Test script')
    WID_BUTTON_runscript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_EXECUTESCRIPT' ,XOFFSET=1024+10+260+110*2 ,YOFFSET=10+40+430+10  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='EXECUTE Script')
    WID_BUTTON_runscript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_EXECUTESCRIPTDARK' ,XOFFSET=1024+10+260+110*2 ,YOFFSET=10+40+430+10+35  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='EXECUTE/no disp.')
    WID_BUTTON_savescript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_SAVESCRIPT' ,XOFFSET=1024+10+260+110*1 ,YOFFSET=10+40+430+10  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Save script')
    WID_BUTTON_savescript = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_LOADSCRIPT' ,XOFFSET=1024+10+260+110*0 ,YOFFSET=10+40+430+10+35  $
      ,SCR_XSIZE=100 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Load script')
  endif



  WID_BUTTON_vieworig = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_VIEWORIG' ,XOFFSET=10 ,YOFFSET=10  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Original Image')
  WID_BUTTON_reset = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_RESET' ,XOFFSET=10 ,YOFFSET=10+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Revert to Original Image')
  WID_BUTTON_gsmooth = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_GSMOOTH' ,XOFFSET=10 ,YOFFSET=10+35*2  $
    ,SCR_XSIZE=87 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Gauss. Filt.')
  WID_BUTTON_msmooth = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_MSMOOTH' ,XOFFSET=100 ,YOFFSET=10+35*2  $
    ,SCR_XSIZE=87 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Median Filt.')
  WID_BUTTON_backsub = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_BACKSUB' ,XOFFSET=10 ,YOFFSET=10+35*3  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Background Subtraction')
  WID_BUTTON_viewgray = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_VIEWGRAY' ,XOFFSET=10 ,YOFFSET=10+35*4  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Working Grayscale Image')


  wid_base_thres = widget_base(WID_BASE_sfa, xoffset=10, yoffset = 10+35*5, scr_xsize = 190, scr_ysize=70)
  wid_slider_thres = cw_fslider(wid_base_thres, /DOUBLE,minimum=0, maximum=65535.,scroll = 1., $
    title='Threshold Value',value = 200.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_THRES',format='(f12.1)')
  WID_BUTTON_thres = Widget_Button(WID_BASE_sfa,  $
    UNAME='WID_BUTTON_THRES' ,XOFFSET=10 ,YOFFSET=10+35*5+80-10  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Segment Image')
  WID_DROPLIST_THRES = Widget_Droplist(WID_BASE_sfa,  $
    UNAME='WID_DROPLIST_THRESHOLD' ,XOFFSET=10,YOFFSET=10+35*5+80+25 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=30 ,VALUE=['Simple Threshold (Slider)','Otsu','Mean','Isodata','Moments','Max Entropy','Min Error','Otsu-Rosin(1:2)','Otsu-Rosin(2:1)',$
    'Otsu-Rosin(slider=%Otsu)','Simple Threshold on masked data']);,'Zamir-Soferman Water algo.'])
  WID_DROPLIST_UPDATE = Widget_Droplist(WID_BASE_sfa,  $
    UNAME='WID_DROPLIST_UPDATE' ,XOFFSET=10,YOFFSET=10+35*5+80+25+30 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=35 ,VALUE=['Overwrite Old Binary','Add to (OR operation)','Keep new blob, Ignore overlap']) ;, 'Zamir-Soferman'])





  WID_BUTTON_viewmask = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_VIEWMASK' ,XOFFSET=10 ,YOFFSET=455  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Binary Mask')
  WID_BUTTON_viewmaskoverlay = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_VIEWMASKOVERLAY' ,XOFFSET=10 ,YOFFSET=455+32  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Binary: Grayscale Overlay')

  ;  WID_BUTTON_clearscreen = Widget_Button(WID_BASE_sfa,  $
  ;    UNAME='WID_BUTTON_HISTOGRAMINTENSITY' ,XOFFSET=10 ,YOFFSET=450+35*2 $
  ;    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Histogram of Pixels')
  ;  WID_DROPLIST_mask = Widget_Droplist(WID_BASE_sfa,  $
  ;    UNAME='WID_DROPLIST_HISTO' ,XOFFSET=10 ,YOFFSET=450+35*3,SCR_XSIZE=180  $
  ;    ,SCR_YSIZE=25 ,TITLE='Option:' ,VALUE=['Entire Image','FA region','Cell region'])
  ;

  WID_DROPLIST_MORPH = Widget_Droplist(WID_BASE_sfa,  $
    UNAME='WID_DROPLIST_MORPH' ,XOFFSET=10,YOFFSET=500+35+50 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=26 ,VALUE=['Grow (Dilate)','Shrink (Erode)','Smooth&grow (Close)','Smooth&shrink (Open)','Thin'])
  WID_BUTTON_morph = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_MORPH' ,XOFFSET=10 ,YOFFSET=500+35+50+25  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Morphological Operation on Mask')

  WID_DROPLIST_MOUSE = Widget_Droplist(WID_BASE_sfa,  $
    UNAME='WID_DROPLIST_MOUSE' ,XOFFSET=10,YOFFSET=500+35+50+25+35*1 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=26 ,VALUE=['Mouse: Inactive','Mouse: pick threshold value','Fill flood add to selection','Fill flood subtraction','Inverse fill flood selection','Remove selected region',$
    'Grow selected region','Shrink selected region','Smooth&grow selected region','Smooth&shrink selected region'])
  ;  WID_DROPLIST_STACK = Widget_Droplist(WID_BASE_sfa,  $
  ;    UNAME='WID_DROPLIST_STACK' ,XOFFSET=10,YOFFSET=500+35+50+25+35*2-5 ,SCR_XSIZE=180  $
  ;    ,SCR_YSIZE=26 ,VALUE=['Current frame','Whole stack'])

  wid_base_min= widget_base(WID_BASE_sfa, xoffset=10, yoffset = 700, scr_xsize = 190, scr_ysize=80)
  wid_slider_min = cw_fslider(wid_base_min, /DOUBLE,minimum=0, maximum=2000.,scroll = 1., $
    title='Minimum Region Size (pixels)',value = 0.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MIN',format='(I7)')
  wid_base_max= widget_base(WID_BASE_sfa, xoffset=10, yoffset = 700+80, scr_xsize = 190, scr_ysize=80)
  wid_slider_max = cw_fslider(wid_base_max, /DOUBLE,minimum=0, maximum=2000.,scroll = 1., $
    title='Maximum Region Size (pixels)',value = 1000.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MAX',format='(I7)')
  WID_BUTTON_size = Widget_Button(WID_BASE_sfa,  $
    UNAME='WID_BUTTON_SIZE' ,XOFFSET=10 ,YOFFSET=700+80+80  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Perform Size Exclusion')
  WID_BUTTON_sizes = Widget_Button(WID_BASE_sfa,  $
    UNAME='WID_BUTTON_SIZES' ,XOFFSET=10 ,YOFFSET=700+80+80+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Exclude Small Regions only')
  WID_BUTTON_fill = Widget_Button(WID_BASE_sfa,  $
    UNAME='WID_BUTTON_FILL' ,XOFFSET=10 ,YOFFSET=700+80+80+35+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Fill Small Holes (use min slider)')

  WID_BUTTON_done = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_DONE' ,XOFFSET=10 ,YOFFSET=960+5+35  $
    ,SCR_XSIZE=87 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='DONE')
  WID_BUTTON_cancel = Widget_Button(wid_base_sfa,  $
    UNAME='WID_BUTTON_CANCEL' ,XOFFSET=100 ,YOFFSET=960+35+5  $
    ,SCR_XSIZE=87 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='CANCEL')

  WID_DROPLIST_color = Widget_Droplist(WID_BASE_sfa,  $
    UNAME='WID_DROPLIST_COLOR' ,XOFFSET=5 ,YOFFSET=435-75+60+30-20 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=25 ,TITLE='Color Scale' ,VALUE=['Grayscale #0', 'Rainbow','#5 Std Gamma II', '#25 Mac Style','Red Temperature # 3','#33 Blue-Red','#34 Rainbow','Inverse Grayscale'])

  ;  WID_BUTTON_revert = Widget_Button(wid_base_sfa,  $
  ;    UNAME='WID_BUTTON_DONE' ,XOFFSET=10 ,YOFFSET=360  $
  ;    ,SCR_XSIZE=87 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='REVERT')
  if stackmode ne 0 then begin
    WID_BUTTON_overlayrevert = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_COMPAREFAMASK' ,XOFFSET=10 ,YOFFSET=360  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Compare old(G) vs. new mask(R)')
    WID_BUTTON_commit = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_COMMIT' ,XOFFSET=10 ,YOFFSET=360+35  $
      ,SCR_XSIZE=50 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Commit')
    WID_BUTTON_commit = Widget_Button(wid_base_sfa,  $
      UNAME='WID_BUTTON_COMMITPROCEED' ,XOFFSET=65 ,YOFFSET=360+35  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Commit and Proceed')
  endif

  oldwindow = !d.window

  Widget_Control, /REALIZE, wid_base_sfa
  XManager, 'WID_BASE_SFA', wid_base_sfa, /NO_BLOCK

  sfawindow = !d.window

end

pro sfadrawevents, event
  common param, pixelsize, timestep, viewmode, colormode, framerate
  ;common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common display, wxsz, wysz, mainwindow, zoomcoord, autoscale, screenmode
  common mouse, mousedown, infoinitial, mousemode
  common interactive, imode, drawwindow, undoimage
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table


  wxsz = 1024. & wysz = 1024.
  if size(properties,/type) ne 8 then return
  width = (size(binaryimage,/dimensions))[0]
  height = (size(binaryimage,/dimensions))[1]
  mgw = wxsz/width
  drawwindow = !d.window
  ;print, 'Event handler'
  ;Fanning's IDL Coyote code: drawbox_widget
  properties.xpixels = width
  properties.ypixels = height

  ;print, event.type
  if imode eq 0 then return
  IF event.type GT 2 THEN RETURN
  eventtypes = ['NONE','RC','LC','MC']
  if (event.type eq 0) and (event.press eq 1) then thisevent = eventtypes[1] else $
    if (event.type eq 0) and (event.press eq 4) then thisevent = eventtypes[2] else $
    if (event.type eq 0) and (event.press eq 2) then thisevent = eventtypes[3] else $
    thisevent =  eventtypes[0]

  dostack = 0;widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_STACK'),/droplist_select)

  ;print, thisevent,dostack
  case thisevent of
    'RC': begin
      datax = event.x/mgw
      datay = event.y/mgw
      ;print, datax, datay, imode
      if ~dostack then case imode of
      1:widget_control,widget_info(event.top,find_by_uname ='WID_SLIDER_THRES'),set_value=grayimage[datax,datay]
      2: binaryimage = binaryimage or getfloodfillimage(grayimage, datax, datay,  grayimage[datax,datay])
      3:  binaryimage = binaryimage and ~getfloodfillimage(grayimage, datax, datay,  grayimage[datax,datay])
      4:binaryimage = binaryimage or ~getfloodfillimage(grayimage, datax, datay, grayimage[datax,datay])
      5:binaryimage = binaryimage xor getselectedregionimage(binaryimage, datax, datay)
      6: binaryimage =  binaryimage or getselectedregionimage(binaryimage, datax, datay,/grow)
      7: binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/shrink),update=3)
      8:  binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/smoothgrow),update=3)
      9:binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/smoothshrink),update=3)
      ;10:   binaryimage=combinebinary(binaryimage,splitregions(getselectedregionimage(binaryimage, datax, datay),grayimage),update= 3)
    endcase

    if dostack then case imode of
    1:widget_control,widget_info(event.top,find_by_uname ='WID_SLIDER_THRES'),set_value=grayimage[datax,datay]
    2: stackprocess, event, datax, datay, imodex = imode, threshold = grayimage[datax,datay]
    3: stackprocess, event, datax, datay, imodex = imode, threshold = grayimage[datax,datay]
    4: stackprocess, event, datax, datay, imodex = imode, threshold = grayimage[datax,datay]
    5: stackprocess, event, datax, datay, imodex = imode
    6: stackprocess, event, datax, datay, imodex = imode
    7: stackprocess, event, datax, datay, imodex = imode
    8: stackprocess, event, datax, datay, imodex = imode
    9: stackprocess, event, datax, datay, imodex = imode
    else:
  endcase

  showmaskoverlay, event
end
else:
endcase
end

function countregion, binaryimage
  label = label_region(binaryimage)
  return, max(label)
end

function regionprops, binaryimage
  ; after matlab regionprops
  label = label_region(binaryimage)
  numregion = max(label)
  if numregion lt 1 then return, -1
  results = replicate({regionprops,label:0L, area:0L,centroid:[0.0,0.0]},numregion)
  for i = 1, numregion do begin
    thislabel = where(label eq i, area)
    results[i-1].area = area
    results[i-1].label = i
  endfor
  return, results
end


function combinebinary, binary1, binary2, update=update
  if ~keyword_set(update) then return, binary2
  case update of
    1: return, binary1 or binary2
    2: begin  ; kill overlap, keep difference
      label1 = label_region(binary1)
      label2 = label_region(binary2)
      results = binary1*0
      numreg1 = max(label1)
      numreg2 = max(label2)
      ; print, numreg1, numreg2
      if numreg1 lt 1 then return, binary2
      if numreg2 lt 1 then return, binary1
      for i = 1, numreg2 do begin  ; if blob in image 2 doesn't overlap with image1 keep this blob
        thislabel = where(label2 eq i, area)
        overlapscore = total(binary1[thislabel]*binary2[thislabel])
        if overlapscore eq 0 then results[thislabel ] = 1
      endfor
      ;print, total(results)
      for j = 1, numreg1 do begin ; if a blob in image 1 overlap with a blob in image 2.. keep the blob in image 1
        thislabel = where(label1 eq j, area)
        overlapscore = total(binary1[thislabel]*binary2[thislabel])
        if overlapscore gt 0 then results[thislabel ] = 1
      endfor
      ;print, total(results)
      return, results
    end
    3: begin  ;keep difference, update overlap
      label1 = label_region(binary1)
      label2 = label_region(binary2)
      results = binary1*0
      numreg1 = max(label1)
      numreg2 = max(label2)
      ; print, numreg1, numreg2
      if numreg1 lt 1 then return, binary2
      if numreg2 lt 1 then return, binary1
      for i = 1, numreg1 do begin  ; if blob in image 1 doesn't overlap with image2 keep the blob
        thislabel = where(label1 eq i, area)
        overlapscore = total(binary1[thislabel]*binary2[thislabel])
        if overlapscore eq 0 then results[thislabel ] = 1
      endfor
      ;print, total(results)
      for j = 1, numreg2 do begin ; if a blob in image 2 overlap with a blob in image 1.. update the blob in image 1
        thislabel = where(label2 eq j, area)
        overlapscore = total(binary1[thislabel]*binary2[thislabel])
        if overlapscore gt 0 then results[thislabel] = 1
      endfor
      ;print, total(results)
      return, results
    end
  endcase
end


function removelargepatch, binaryimage, maxpatchsize
  width = (size(binaryimage,/dimensions))[0]
  height = (size(binaryimage,/dimensions))[1]
  padimage = bytarr(width+2, height+2)
  padimage[1:width,1:height] = binaryimage
  label = label_region(padimage)
  prop = regionprops(padimage)
  if size(prop, /type) ne 8 then return,binaryimage

  rejectarea = where(prop.area gt maxpatchsize, numreject)

  if numreject gt 0 then begin
    for i = 0, numreject-1 do begin
      rem = where(label eq prop[rejectarea[i]].label)
      padimage[rem] =0
    endfor
  endif else return, binaryimage
  return, padimage[1:width,1:height]
end

function fillsmallholes, binaryimage, minholesize

  inverseimage = ~binaryimage
  holeremoved = removesmallpatch(inverseimage, minholesize)
  holes = holeremoved xor inverseimage
  return, binaryimage xor holes
end


function removesmallpatch, binaryimage, minpatchsize, biggest=biggest
  width = (size(binaryimage,/dimensions))[0]
  height = (size(binaryimage,/dimensions))[1]
  padimage = bytarr(width+2, height+2)
  padimage[1:width,1:height] = binaryimage
  label = label_region(padimage)
  prop = regionprops(padimage)
  if size(prop, /type) ne 8 then return,binaryimage
  rejectarea = where(prop.area lt minpatchsize, numreject)

  if keyword_set(biggest) then begin
    maxi = 0
    maxval = max(prop.area, maxi)
    ;print, maxi
    maxlbl = prop[maxi].label
    getindex = where(label eq maxlbl, count)
    ;print, count
    cellmask = padimage*0
    cellmask[getindex] = 1
    return, cellmask[1:width,1:height]
  endif

  if numreject gt 0 then begin
    for i = 0, numreject-1 do begin
      rem = where(label eq prop[rejectarea[i]].label)
      padimage[rem] =0
    endfor
  endif else return, binaryimage
  return, padimage[1:width,1:height]
end