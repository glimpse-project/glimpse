function getURLParameter(name) {
  return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search) || [null, ''])[1].replace(/\+/g, '%20')) || null;
}

function httpGetSync(path)
{
    var httpReq = new XMLHttpRequest();
    httpReq.open("GET", path, false); // (synchronous)
    httpReq.send(null);
    return httpReq.responseText;
}

// Just to be consistent with Python...
Math.radians = function(degrees) {
  return degrees * Math.PI / 180;
};
 
Math.degrees = function(radians) {
  return radians * 180 / Math.PI;
};

var EventTarget = function() {
    this.listeners = {};
};

EventTarget.prototype.listeners = null;
EventTarget.prototype.removeEventListener = function(type, callback) {
    if(!(type in this.listeners)) {
        return;
    }
    var stack = this.listeners[type];
    for(var i = 0, l = stack.length; i < l; i++){
        if(stack[i] === callback){
            stack.splice(i, 1);
            return this.removeEventListener(type, callback);
        }
    }
};

EventTarget.prototype.addEventListener = function(type, callback) {
    if(!(type in this.listeners)) {
        this.listeners[type] = [];
    }
    this.removeEventListener(type, callback);
    this.listeners[type].push(callback);
};

EventTarget.prototype.dispatchEvent = function(event){
    if(!(event.type in this.listeners)) {
        return;
    }
    var stack = this.listeners[event.type];
    event.target = this;
    for(var i = 0, l = stack.length; i < l; i++) {
        stack[i].call(this, event);
    }
};

EventTarget.prototype.on = function(type, callback) {
    this.addEventListener(type, callback);
}

EventTarget.prototype.once = function(type, callback) {
  function _once_wraper() {
    this.removeListener(type, _once_wrapper);
    return callback.apply(this, arguments);
  }
  return this.on(type, _once_wrapper);
};

function LoadPNG(path) {
    var req = new EventTarget();

    var httpReq = new XMLHttpRequest();
    httpReq.open("GET", path, true);
    httpReq.responseType = "arraybuffer";

    httpReq.onload = function (oEvent) {
        var arrayBuffer = httpReq.response; // Note: not httpReq.responseText
        if (arrayBuffer) {
            var byteArray = new Uint8ClampedArray(arrayBuffer);
            var img = new png.PNG({ filterType:4 }).parse(byteArray, (error, data) => {
                if (error) {
                    var ev = { type: 'error', message: error };
                    req.dispatchEvent(ev);
                } else {
                    var ev = { type: 'load', data: data };
                    req.dispatchEvent(ev);
                }
            });
        }
    };

    httpReq.send(null);

    return req;
}

function LoadPFM(path) {
    var req = new EventTarget();

    var httpReq = new XMLHttpRequest();
    httpReq.open("GET", path, true);
    httpReq.responseType = "arraybuffer";

    httpReq.onload = function (oEvent) {
        var arrayBuffer = httpReq.response; // Note: not httpReq.responseText
        if (arrayBuffer) {
            var uint8Array = new Uint8Array(arrayBuffer);

            var nl = "\n".charCodeAt(0);
            var i = 0;
            for (i = 0; i < uint8Array.length; i++) {
                if (uint8Array[i] === nl)
                    break;
            }
            if (i === uint8Array.length) {
                var ev = { type: 'error', message: "Missing PFM header field" };
                req.dispatchEvent(ev);
                return;
            }
            var header_end = i;
            var header_u8 = new Uint8Array(arrayBuffer, 0, header_end);
            
            var dim_start = header_end + 1;
            for (i = dim_start; i < uint8Array.length; i++) {
                if (uint8Array[i] === nl)
                    break;
            }
            if (i === uint8Array.length) {
                var ev = { type: 'error', message: "Missing PFM dimensions field" };
                req.dispatchEvent(ev);
                return;
            }
            var dim_end = i;
            var dim_u8 = new Uint8Array(arrayBuffer, dim_start, dim_end - dim_start);

            var scale_start = dim_end + 1;
            for (i = scale_start; i < uint8Array.length; i++) {
                if (uint8Array[i] === nl)
                    break;
            }
            if (i === uint8Array.length) {
                var ev = { type: 'error', message: "Missing PFM scale field" };
                req.dispatchEvent(ev);
                return;
            }
            var scale_end = i;
            var scale_u8 = new Uint8Array(arrayBuffer, scale_start, scale_end - scale_start);

            var decoder = new TextDecoder('utf-8');
            var header = decoder.decode(header_u8);
            var dim = decoder.decode(dim_u8);
            var scale = parseFloat(decoder.decode(scale_u8));

            if (header === "Pf")
                var n_components = 1;
            else if (header === "PF")
                var n_components = 3;
            else {
                var ev = { type: 'error', message: "Missing PFM header of \"Pf\" or \"PF\"" };
                req.dispatchEvent(ev);
                return;
            }

            var dimensions = dim.split(" ");
            var width = parseInt(dimensions[0], 10);
            var height = parseInt(dimensions[1], 10);

            if (scale !== -1) {
                var ev = { type: 'error', message: "Only handles PFMs with little endian scale of -1.0" };
                req.dispatchEvent(ev);
                return;
            }

            //console.log("PFM header: \"" + header + "\"");
            //console.log("PFM dim: " + width + "x" + height);
            //console.log("PFM scale: " + scale);

            var remainder = uint8Array.length - (scale_end + 1);
            var expected = width * height * 4 * n_components;

            if (remainder !== expected) {
                var ev = { type: 'error', message: "Expected " + expected + " bytes of data but found " + remainder };
                req.dispatchEvent(ev);
                return;
            }

            /* The float data needs to be copied to a separate buffer to ensure
             * it is properly aligned
             */
            var aligned_buf = new ArrayBuffer(remainder);
            var data_u8 = new Uint8Array(aligned_buf);
            for (i = 0; i < remainder; i++)
                data_u8[i] = uint8Array[i + scale_end + 1];

            var float32Array = new Float32Array(aligned_buf);
            var pfm = {
                width: width,
                height: height,
                n_components: n_components,
                data: float32Array,
            };
            var ev = { type: 'load', data: pfm };
            req.dispatchEvent(ev);
        }
    };

    httpReq.send(null);

    return req;
}

function depth_image_to_point_cloud(depth, vfov) {
    var depth_width = depth.width;
    var depth_height = depth.height;

    var depth_half_width = depth_width / 2;
    var depth_half_height = depth_height / 2;

    var aspect = depth_width / depth_height;

    console.log("depth_image_to_point_cloud:")
    console.log("width = " + depth_width + ", height=" + depth_height + ", vertical fov = " + vfov + ", w/h aspect = " + aspect);
    
    var tan_half_vfov = Math.tan(Math.radians(vfov) / 2);
    
    var tan_half_hfov = tan_half_vfov * aspect;
    var hfov = Math.atan(tan_half_hfov) * 2;
    
    console.log("implied h fov = " + Math.degrees(hfov));
    
    var point_cloud = new Float32Array(depth_width * depth_height * 3 * 4);
    var n_points = 0;

    for (var y = 0; y < depth_height; y++) {
        for (var x = 0; x < depth_width; x++) {
            var z = depth.data[depth_width * y + x];
            if (z === 0) {
                console.log("skipping zero depth value");
                continue;
            }
            if (!Number.isFinite(z)) {
                console.log("skipping INF depth value");
                continue;
            }
                
            var half_field_width = tan_half_hfov * z;
            var half_field_height = tan_half_vfov * z;

            /* s,t are normalized image coordinates with (0,0) in the center
             * ranging [-1,1], left to right and top to bottom
             */
            var s = (x / depth_half_width) - 1.0;
            var t = (y / depth_half_height) - 1.0;

            var x_ = half_field_width * s;
            var y_ = half_field_height * t;
            
            //console.log("x=" + x_ + ",y=" + y_+ ",z=" + z + " (s=" + s + ", t=" + t + ")");

            var pos = n_points * 3;
            point_cloud[pos+0] = x_;
            point_cloud[pos+1] = y_;
            point_cloud[pos+2] = z;
            n_points++;
        }
    }

    return new Float32Array(point_cloud, 0, n_points * 3 * 4);
}

function point_cloud_to_image(point_cloud, vfov, width, height) {
    var depth = new Float32Array(width * height);

    var half_width = width / 2;
    var half_height = height / 2;
    
    var tan_half_vfov = Math.tan(Math.radians(vfov) / 2);
    
    var aspect = width / height;
    
    console.log("point_cloud_to_image:");
    console.log("width = " + width + ", height=" + height + ", vertical fov = " + vfov + ", w/h aspect = " + aspect);

    var tan_half_hfov = tan_half_vfov * aspect;
    var hfov = Math.atan(tan_half_hfov) * 2;
    
    console.log("implied h fov = " + Math.degrees(hfov));

    for (var i = 0; i < point_cloud.length; i += 3) {
        var half_field_width = tan_half_hfov * z;
        var half_field_height = tan_half_vfov * z;

        var ndc_point_x = x / half_field_width;
        var ndc_point_y = y / half_field_height;

        var x_ = Math.round((ndc_point_x + 1.0) * half_width);
        var y_ = Math.round((ndc_point_y + 1.0) * half_height);

        //console.log("x=" + x_ + ",y=" + y_ + ",z=" + z + " (ndc_x=" + ndc_point_x + ", ndc_y=" + ndc_point_y + ")");
        depth[y_ * width + x_] = z;
    }
        
    return depth;
}

function point_cloud_to_image_coords(point_cloud, vfov, width, height) {
    var out = new Float32Array(width * height * 2);

    var half_width = width / 2;
    var half_height = height / 2;
    
    var tan_half_vfov = Math.tan(Math.radians(vfov) / 2);
    
    var aspect = width / height;
    
    console.log("point_cloud_to_image_coords:");
    console.log("width = " + width + ", height=" + height + ", vertical fov = " + vfov + ", w/h aspect = " + aspect);

    var tan_half_hfov = tan_half_vfov * aspect;
    var hfov = Math.atan(tan_half_hfov) * 2;
    
    console.log("implied h fov = " + Math.degrees(hfov));

    for (var i = 0; i < point_cloud.length; i += 3) {
        var half_field_width = tan_half_hfov * z;
        var half_field_height = tan_half_vfov * z;

        var ndc_point_x = x / half_field_width;
        var ndc_point_y = y / half_field_height;

        var x_ = Math.round((ndc_point_x + 1.0) * half_width);
        var y_ = Math.round((ndc_point_y + 1.0) * half_height);

        //console.log("x=" + x_ + ",y=" + y_ + ",z=" + z + " (ndc_x=" + ndc_point_x + ", ndc_y=" + ndc_point_y + ")");
        var pos = (y_ * width + x_) * 2;
        out[pos+0] = x_;
        out[pos+0] = y_;
    }
        
    return out;
}

function reproject_depth(depth, depth_yfov, width, height, yfov) {
    var point_cloud = depth_image_to_point_cloud(depth, depth_yfov);
    return point_cloud_to_image(point_cloud, yfov, width, height);
}

function init_color_stops() {
    var rainbow = [
        [[1, 1, 0], "yellow"],
        [[0, 0, 1], "blue"],
        [[0, 1, 0], "green"],
        [[1, 0, 0], "red"],
        [[0, 1, 1], "cyan"],
        //[[1, 0, 1], "magenta"]
    ];

    var stops = [];
    var r = 0;
    var f = 1.0;
    for (var i = 0; i < 5000; i += 250) {
        var dist = i / 1000;
        var c = rainbow[r][0];

        stops.push([dist, [c[0]*f, c[1]*f, c[2]*f], f, rainbow[r][1]]);
        r++;
        if (r >= rainbow.length) {
            f *= 0.8;
            r -= rainbow.length;
        }
            
        r = (r + 1) % rainbow.length;
    }
    
    console.log("stops:")
    for (var i = 0; i < stops.length; i++) {
        var s = stops[i];
        console.log(s[0] + "m = " + s[2] + " x " + s[3]);
    }

    return stops;
}

function get_color_for_depth(depth, stops) {
    var color = [ 255, 255, 255, 255 ];

    for (var i = 1; i < stops.length; i++) {
        if (depth < stops[i][0]) {
            var t = (depth - stops[i-1][0]) / (stops[i][0] - stops[i-1][0]);

            var col0 = stops[i-1][1];
            var col1 = stops[i][1];

            var r = (1 - t) * col0[0] + t * col1[0];
            var g = (1 - t) * col0[1] + t * col1[1];
            var b = (1 - t) * col0[2] + t * col1[2];

            color[0] = r;
            color[1] = g;
            color[2] = b;

            return color;
        }
    }

    return color;
}

/*
 * 
 */
function colorize_depth(depth, color_stops) {
    var width = depth.width;
    var height = depth.height;

    var out = new Uint8ClampedArray(width * height * 4);
    for (var y = 0; y < height; y++) {
        for (var x = 0; x < width; x++) {
            var depth_pos = width * y + x;
            var pos = depth_pos * 4;

            var color = get_color_for_depth(depth.data[depth_pos], color_stops);

            out[pos + 0] = color[0] * 255;
            out[pos + 1] = color[1] * 255;
            out[pos + 2] = color[2] * 255;
            out[pos + 3] = color[3] * 255;
        }
    }

    return { width: width, height: height, data: out };
}

function color_point_cloud_with_stops(point_cloud, color_stops) {
    var point_colors = new Float32Array(point_cloud.length);

    for (var i = 0; i < point_cloud.length; i += 3) {
        var z = point_cloud[i+2];
        var color = get_color_for_depth(z, color_stops);

        point_colors[i+0] = color[0];
        point_colors[i+1] = color[1];
        point_colors[i+2] = color[2];
    }

    return point_colors;
}

var color_stops = init_color_stops();
var data_dir = getURLParameter("dir");
var controllers = [];
var views = [];
var obj_world_scale = 100;
var initial_frame_no = getURLParameter("frame");

var index = httpGetSync(data_dir + "/index");
index = index.split('\n');

$("#index-info").html(index.length + " frames indexed");


function init_view() {

    var view = {};

    var renderer = new THREE.WebGLRenderer({canvas: $("#training-canvas")[0]});
    view.renderer = renderer;

    var scene = new THREE.Scene();
    view.scene = scene;

    scene.background = new THREE.Color(0xcccccc);

    camera = new THREE.PerspectiveCamera(60, renderer.getSize().width / renderer.getSize().height, 1, 1000);
    view.camera = camera;

    camera.position.z = 500;
    scene.add(camera);

    var light = new THREE.DirectionalLight( 0xffffff );
    light.position.set( 0, 1, 1 ).normalize();
    scene.add(light);

    controls = new THREE.TrackballControls(camera, renderer.domElement);
    view.controls = controls;

    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;

    controls.noZoom = false;
    controls.noPan = false;

    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.3;

    controls.keys = [ 65, 83, 68 ];

    controls.addEventListener('change', render);

    var pointLight = new THREE.PointLight(0xFFFFFF);
    scene.add(pointLight);

    var ambientLight = new THREE.AmbientLight(0x80808080);
    scene.add(ambientLight);

    /*
    var geometry = new THREE.CylinderGeometry(0, 10, 30, 4, 1);
    var material = new THREE.MeshPhongMaterial({ color: 0xffffff, flatShading: true });
    for (var i = 0; i < 500; i++) {
        var mesh = new THREE.Mesh( geometry, material );
        mesh.position.x = ( Math.random() - 0.5 ) * 1000;
        mesh.position.y = ( Math.random() - 0.5 ) * 1000;
        mesh.position.z = ( Math.random() - 0.5 ) * 1000;
        mesh.updateMatrix();
        mesh.matrixAutoUpdate = false;
        scene.add( mesh );
    }
    */


    return view;
}

function load_training_frame(view, frame_no) {
    var frame_path = index[frame_no];


    console.log("Loading frame " + frame_path);

    var frame_meta_json = httpGetSync(data_dir + "/labels" + frame_path + ".json");
    var frame_meta = JSON.parse(frame_meta_json);

    var render_vfov = frame_meta.camera.vertical_fov;

    $("#training-frame-description").html("Frame: " + frame_path + " (" + frame_no + ")<br>Camera fov:" + render_vfov);

    var req = LoadPNG(data_dir + "/labels" + frame_path + ".png");
    req.on('load', (ev) => {
        console.log("loaded PNG " + ev.data.width + "x" + ev.data.height);

        //var canvas = document.getElementsByTagName('canvas')[0];
        //var ctx = canvas.getContext('2d');

        //var clamped = new Uint8ClampedArray(ev.data.data);
        //var img = new ImageData(clamped, ev.data.width, ev.data.height);
        //ctx.putImageData(img, 0, 0);
        var label_tex = new THREE.DataTexture(ev.data.data,
            ev.data.width, ev.data.height, THREE.RGBAFormat, THREE.UnsignedByteType);
        label_tex.needsUpdate = true;
        label_tex.flipY = true;
        var material = new THREE.MeshPhongMaterial({map: label_tex});
        material.side = THREE.DoubleSide;
        //var material = new THREE.MeshPhongMaterial( { map: label_tex, ambient: 0x050505, color: 0x0033ff, specular: 0x555555, shininess: 30  } );
        //var rect = new THREE.CubeGeometry(20, 20, 20);
        var rect = new THREE.PlaneGeometry(ev.data.width, ev.data.height);
        var mesh = new THREE.Mesh(rect, material);
        mesh.position.x = 0;
        mesh.position.y = 0;
        mesh.position.z = 0;

        //view.scene.add(mesh);

        render();
    });
    req.on('error', (ev) => {
        console.error("Failed to load PNG: " +  ev.message);
    });


    req = LoadPFM(data_dir + "/depth" + frame_path + ".pfm");
    req.on('load', (ev) => {
        console.log("loaded PFM " + ev.data.width + "x" + ev.data.height);
        var img = colorize_depth(ev.data, color_stops);

        //var canvas = document.getElementsByTagName('canvas')[0];
        //var ctx = canvas.getContext('2d');
        
        //var id = new ImageData(img.data, img.width, img.height);
        //ctx.putImageData(id, 0, 0);

        var depth_tex = new THREE.DataTexture(img.data,
            ev.data.width, ev.data.height, THREE.RGBAFormat, THREE.UnsignedByteType);
        depth_tex.needsUpdate = true;
        depth_tex.flipY = true;
        //var material = new THREE.Material({map: label_tex});
        var material = new THREE.MeshPhongMaterial( { map: depth_tex } );
        material.side = THREE.DoubleSide;
        var rect = new THREE.PlaneGeometry(ev.data.width/10, ev.data.height/10);
        var mesh = new THREE.Mesh(rect, material);

        mesh.position.x = 0;
        mesh.position.y = 0;
        mesh.position.z = ev.data.width / 10;

        var camera = new THREE.PerspectiveCamera(render_vfov,
                                ev.data.width / ev.data.height,
                                0.1, ev.data.width / 10);
        var helper = new THREE.CameraHelper(camera);

        helper.scale.z = -1;
        helper.updateMatrix();
        helper.matrixAutoUpdate = false;
        view.scene.add(helper);

        view.scene.add(mesh);

        for (var i = 0; i < 5; i++) {
            var marker = new THREE.PlaneGeometry(10, 10);
            var material = new THREE.MeshPhongMaterial( {
                ambient: 0x050505, color: 0x000000,
                specular: 0x555555, shininess: 30  } );
            material.side = THREE.DoubleSide;
            var mesh = new THREE.Mesh(marker, material);
            mesh.position.x = 10;
            mesh.position.z = i * obj_world_scale;

            view.scene.add(mesh);

            for (var j = 1; j < 10; j++) {
                var material = new THREE.MeshPhongMaterial( {
                    ambient: 0x050505, color: 0x000000,
                    specular: 0x555555, shininess: 30  } );
                material.side = THREE.DoubleSide;
                var mesh = new THREE.Mesh(marker, material);
                mesh.position.z = (i + (j / 10)) * obj_world_scale;
                mesh.position.x = 10;
                mesh.scale.x = 0.5;
                mesh.scale.y = 0.5;
                mesh.scale.z = 0.5;
                view.scene.add(mesh);
            }
        }

        var point_cloud = depth_image_to_point_cloud(ev.data, render_vfov);
        var n_points = point_cloud.length / 3;
        var geometry = new THREE.BufferGeometry();
        geometry.addAttribute('position', new THREE.BufferAttribute(point_cloud, 3));

        var point_colors = color_point_cloud_with_stops(point_cloud, color_stops);
        geometry.addAttribute('color_in', new THREE.BufferAttribute(point_colors, 3));


/*
        for (var i = 0; i < point_cloud.length; i += 3) {
            var x = point_cloud[i];
            var y = point_cloud[i+1];
            var z = point_cloud[i+2];

            var point = new THREE.PlaneGeometry(1, 1);
            var material = new THREE.MeshPhongMaterial( {
                ambient: 0x050505, color: 0x0033ff,
                specular: 0x555555, shininess: 30  } );
            material.side = THREE.DoubleSide;
            var mesh = new THREE.Mesh(point, material);

            point.position.x = x;
            point.position.y = y;
            point.position.z = z;

            view.scene.add(mesh);
        }
        */

        uniforms = {
        };
        var shaderMaterial = new THREE.ShaderMaterial( {

            uniforms:       uniforms,
            vertexShader:   document.getElementById( 'vertexshader' ).textContent,
            fragmentShader: document.getElementById( 'fragmentshader' ).textContent,

            blending:       THREE.AdditiveBlending,
            depthTest:      false,
            transparent:    false
        });

        //var material = new THREE.MeshPhongMaterial( {
        //    ambient: 0x050505, color: 0x0033ff,
        //    specular: 0x555555, shininess: 30  } );
        //material.side = THREE.DoubleSide;
        var particles = new THREE.Points(geometry, shaderMaterial);
        particles.scale.x = obj_world_scale;
        particles.scale.y = -obj_world_scale;
        particles.scale.z = obj_world_scale;

        particles.updateMatrix();
        particles.matrixAutoUpdate = false;

        //var mesh = new THREE.Mesh(point, material);
        view.scene.add(particles);

        render();
    });
    req.on('error', (ev) => {
        console.error("Failed to load PFM: " +  ev.message);
    });

    var material = new THREE.MeshPhongMaterial({ color: 0x00ff00, flatShading: true });

    var geometry = new THREE.IcosahedronBufferGeometry(1, 5);
    for (var i = 0; i < frame_meta.bones.length; i++) {
        var bone = frame_meta.bones[i];

        var mesh = new THREE.Mesh(geometry, material);
        mesh.position.x = bone.head[0] * obj_world_scale;
        mesh.position.y = bone.head[1] * obj_world_scale;
        mesh.position.z = -bone.head[2] * obj_world_scale;
        view.scene.add(mesh);

        mesh = new THREE.Mesh(geometry, material);
        mesh.position.x = bone.tail[0] * obj_world_scale;
        mesh.position.y = bone.tail[1] * obj_world_scale;
        mesh.position.z = -bone.tail[2] * obj_world_scale;
        view.scene.add(mesh);
    }
}

function animate() {
    requestAnimationFrame(animate);

    for (var i = 0; i < views.length; i++) {
        var view = views[i];
        view.controls.update();
    }
}

function render() {
    for (var i = 0; i < views.length; i++) {
        var view = views[i];
        view.renderer.render(view.scene, view.camera);
    }
}


var training_view = init_view();
$("#training-view").append(training_view.renderer.domElement);
views.push(training_view);

$("#training-frame-slider").slider({
    range: "max",
    min: 0,
    max: index.length - 1,
    value: 0,
    slide: function( event, ui ) {
        //$("#current-frame-no").html(ui.value);
        load_training_frame(training_view, ui.value);
    }
});
if (initial_frame_no < index.length)
    load_training_frame(training_view, initial_frame_no);
else
    load_training_frame(training_view, 0);


function onWindowResize() {
    for (var i = 0; i < views.length; i++) {
        var view = views[i];

        var width = view.renderer.domElement.clientWidth;
        var height = view.renderer.domElement.clientHeight;

        view.camera.aspect = width / height;
        view.camera.updateProjectionMatrix();

        view.renderer.setSize(width, height, false);

        view.controls.handleResize();
    }

    render();
}
window.addEventListener('resize', onWindowResize, false );
onWindowResize();

animate();



