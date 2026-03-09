/**
 * SuperTransformer.ts (Final Stable Version)
 * GPU-процессор: аналитическая векторизация края (fwidth), 
 * линейная трапеция без искажений и свободные границы (no-clipping).
 */

export const DEFAULT_GPU_EXPAND_FACTOR = 3;

export interface TransformerParams {
    mode?: number;    // 0: Taper, 1: Parallel, 2: Rotate/Scale, 3: Identity (pass-through for stroke)
    angle?: number;   // Угол (в градусах)
    force?: number;   // Сила деформации
    offset?: number;  // Смещение
    scale?: number;   // Масштаб
    expandViewport?: boolean; // Expand render viewport to capture overflow
    expandFactor?: number; // Canvas/view expansion multiplier (e.g. 3 => 10px -> 30px)
    // Stroke/Border rendering
    strokeWidth?: number;    // stroke width in screen pixels
    strokeColor?: [number, number, number]; // rgb 0..1
    strokeOpacity?: number;  // 0..1
}

export interface PatchRect { x: number; y: number; w: number; h: number; }

export class SuperTransformer {
    private canvas: HTMLCanvasElement;
    private gl: WebGLRenderingContext;
    private program: WebGLProgram;
    private texture: WebGLTexture;
    private buffer: WebGLBuffer;

    constructor() {
        this.canvas = document.createElement('canvas');
        const context = this.canvas.getContext('webgl', { 
            alpha: true, 
            antialias: true,
            premultipliedAlpha: false 
        });

        if (!context) throw new Error("WebGL context failed.");
        this.gl = context;

        // Включаем производные для идеально острого края (fwidth)
        this.gl.getExtension('OES_standard_derivatives');

        this.program = this._initShader();
        this.texture = this._initTexture();
        this.buffer = this._initBuffer();
    }

    private _initShader(): WebGLProgram {
        const gl = this.gl;
        const vs = `attribute vec2 p; varying vec2 v; void main(){ v=p*0.5+0.5; gl_Position=vec4(p,0,1); }`;
        const fs = `
            #extension GL_OES_standard_derivatives : enable
            precision highp float;
            varying vec2 v;
            uniform sampler2D t;
            uniform vec2 res;
            uniform int mode; 
            uniform float a, f, o, s; 
            uniform vec4 win;
            // Stroke uniforms
            uniform float strokeW;
            uniform vec3 strokeRGB;
            uniform float strokeOp;

            void main() {
                vec2 gPos = v * win.zw + win.xy;
                vec2 p = gPos - 0.5;
                p.y = -p.y;

                float aspect = max(win.z, 0.0001) / max(win.w, 0.0001);
                vec2 pA = vec2(p.x * aspect, p.y);

                float r = radians(a);
                vec2 dir = normalize(vec2(cos(r), sin(r)));
                vec2 perp = vec2(-dir.y, dir.x);

                float dAlong = dot(pA, dir);
                float dSide = dot(pA, perp);

                float k = f * 0.01;
                vec2 uv;

                if (mode == 0) { // TAPER (Трапеция)
                    float maxProj = length(vec2(0.5 * aspect, 0.5));
                    float alongNorm = dAlong / max(maxProj, 0.001);
                    float widthScale = clamp(1.0 + alongNorm * k, 0.15, 8.0);

                    float srcAlong = dAlong;
                    float srcSide = dSide / widthScale;
                    vec2 warped = dir * srcAlong + perp * srcSide;
                    uv = vec2(warped.x / aspect, warped.y) + 0.5;
                } 
                else if (mode == 1) { // PARALLEL (Сдвиг)
                    float srcAlong = dAlong - dSide * k;
                    float srcSide = dSide;
                    vec2 warped = dir * srcAlong + perp * srcSide;
                    uv = vec2(warped.x / aspect, warped.y) + 0.5;
                }
                else if (mode == 2) { // ROTATE / SCALE
                    mat2 rotMat = mat2(cos(r), sin(r), -sin(r), cos(r));
                    vec2 warped = (rotMat * pA) / max(s, 0.001);
                    uv = vec2(warped.x / aspect, warped.y) + 0.5;
                }
                else { // Mode 3: IDENTITY (pass-through for stroke)
                    uv = vec2(gPos.x, 1.0 - gPos.y);
                }

                vec4 tex = texture2D(t, uv);
                float bounds = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);

                float edge = fwidth(tex.a);
                float mask = smoothstep(0.5 - edge, 0.5 + edge, tex.a);

                // Stroke rendering: dilate alpha and draw border
                float hasStroke = step(0.001, strokeW);
                vec2 uvD = fwidth(uv);
                float dX = uvD.x * strokeW;
                float dY = uvD.y * strokeW;

                // Sample 8 neighbors to compute dilated alpha
                float dilated = tex.a;
                dilated = max(dilated, texture2D(t, uv + vec2(dX, 0.0)).a);
                dilated = max(dilated, texture2D(t, uv - vec2(dX, 0.0)).a);
                dilated = max(dilated, texture2D(t, uv + vec2(0.0, dY)).a);
                dilated = max(dilated, texture2D(t, uv - vec2(0.0, dY)).a);
                dilated = max(dilated, texture2D(t, uv + vec2(dX * 0.707, dY * 0.707)).a);
                dilated = max(dilated, texture2D(t, uv - vec2(dX * 0.707, dY * 0.707)).a);
                dilated = max(dilated, texture2D(t, uv + vec2(-dX * 0.707, dY * 0.707)).a);
                dilated = max(dilated, texture2D(t, uv + vec2(dX * 0.707, -dY * 0.707)).a);

                float dilatedEdge = fwidth(dilated);
                float dilatedMask = smoothstep(0.5 - dilatedEdge, 0.5 + dilatedEdge, dilated);
                float strokeZone = dilatedMask * (1.0 - mask) * bounds;

                // Composite: shape over stroke
                float shapeA = mask * bounds;
                float strokeA = strokeZone * strokeOp * hasStroke;
                float finalA = shapeA + strokeA * (1.0 - shapeA);
                float safeA = max(finalA, 0.0001);
                vec3 finalRGB = (tex.rgb * shapeA + strokeRGB * strokeA * (1.0 - shapeA)) / safeA;

                gl_FragColor = vec4(finalRGB, finalA);
            }
        `;

        const prog = gl.createProgram()!;
        const add = (type: number, src: string) => {
            const sh = gl.createShader(type)!;
            gl.shaderSource(sh, src);
            gl.compileShader(sh);
            if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
            gl.attachShader(prog, sh);
        };
        add(gl.VERTEX_SHADER, vs);
        add(gl.FRAGMENT_SHADER, fs);
        gl.linkProgram(prog);
        return prog;
    }

    private _initTexture(): WebGLTexture {
        const gl = this.gl;
        const tex = gl.createTexture()!;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, 10242, 33071); // CLAMP_TO_EDGE
        gl.texParameteri(gl.TEXTURE_2D, 10243, 33071); // CLAMP_TO_EDGE
        gl.texParameteri(gl.TEXTURE_2D, 10240, 9729);  // LINEAR
        gl.texParameteri(gl.TEXTURE_2D, 10241, 9729);  // LINEAR
        return tex;
    }

    private _initBuffer(): WebGLBuffer {
        const gl = this.gl;
        const b = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
        return b;
    }

    public render(
        img: HTMLImageElement | HTMLCanvasElement, 
        params: TransformerParams, 
        patch: PatchRect | null = null,
        fullSize: { w: number, h: number } | null = null,
        overrideDpr?: number
    ): HTMLCanvasElement {
        const gl = this.gl;
        const { mode = 0, angle = 0, force = 0, offset = 0, scale = 1 } = params;

        // Expand only when explicitly requested (prevents cumulative growth on chained passes)
        const expand = params.expandViewport
            ? Math.max(1, params.expandFactor ?? DEFAULT_GPU_EXPAND_FACTOR)
            : 1;

        const dpr = overrideDpr ?? (window.devicePixelRatio || 1);
        const baseW = patch ? patch.w : img.width;
        const baseH = patch ? patch.h : img.height;

        this.canvas.width = Math.max(1, Math.round(baseW * expand * dpr));
        this.canvas.height = Math.max(1, Math.round(baseH * expand * dpr));
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        gl.useProgram(this.program);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);

        // View window mapping for shader sampling
        let winArr: number[];
        if (patch && fullSize) {
            winArr = [patch.x / fullSize.w, patch.y / fullSize.h, patch.w / fullSize.w, patch.h / fullSize.h];
        } else if (params.expandViewport) {
            // Center 1x source inside expanded canvas (e.g. 3x => offset -1.0, size 3.0)
            const off = (1 - expand) / 2;
            winArr = [off, off, expand, expand];
        } else {
            winArr = [0, 0, 1, 1];
        }

        const pLoc = gl.getAttribLocation(this.program, "p");
        gl.enableVertexAttribArray(pLoc);
        gl.vertexAttribPointer(pLoc, 2, gl.FLOAT, false, 0, 0);

        gl.uniform1i(gl.getUniformLocation(this.program, "mode"), mode);
        gl.uniform1f(gl.getUniformLocation(this.program, "a"), angle);
        gl.uniform1f(gl.getUniformLocation(this.program, "f"), force);
        gl.uniform1f(gl.getUniformLocation(this.program, "o"), offset);
        gl.uniform1f(gl.getUniformLocation(this.program, "s"), scale);
        gl.uniform4fv(gl.getUniformLocation(this.program, "win"), new Float32Array(winArr));
        gl.uniform2f(gl.getUniformLocation(this.program, "res"), this.canvas.width, this.canvas.height);

        // Stroke uniforms
        const strokeWidth = params.strokeWidth ?? 0;
        const strokeColor = params.strokeColor ?? [1, 1, 1];
        const strokeOpacity = params.strokeOpacity ?? 1;
        gl.uniform1f(gl.getUniformLocation(this.program, "strokeW"), strokeWidth);
        gl.uniform3fv(gl.getUniformLocation(this.program, "strokeRGB"), new Float32Array(strokeColor));
        gl.uniform1f(gl.getUniformLocation(this.program, "strokeOp"), strokeOpacity);

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

        return this.canvas;
    }

    /**
     * Helper: Convert HSL string to RGB 0..1 array for GPU uniform
     */
    static hslToRgb01(hsl: string): [number, number, number] {
        const m = hsl.match(/hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%?\s*,\s*([\d.]+)%?\s*\)/);
        if (!m) return [1, 1, 1];
        const h = parseFloat(m[1]) / 360;
        const s = parseFloat(m[2]) / 100;
        const l = parseFloat(m[3]) / 100;
        if (s === 0) return [l, l, l];
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1/6) return p + (q - p) * 6 * t;
            if (t < 1/2) return q;
            if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
            return p;
        };
        return [hue2rgb(p, q, h + 1/3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1/3)];
    }
}
