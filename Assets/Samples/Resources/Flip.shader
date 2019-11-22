Shader "Hidden/TFLite/Flip"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            int _FlipY;
            int _FlipX;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                
                if (_FlipX > 0)
                {
                    o.uv.x = 1.0 - o.uv.x;
                }
                if (_FlipY > 0)
                {
                    o.uv.y = 1.0 - o.uv.y;
                }

                return o;
            }

            sampler2D _MainTex;

            fixed4 frag (v2f i) : SV_Target
            {
                float2 uv = i.uv;
                return tex2D(_MainTex, uv);
            }
            ENDCG
        }
    }
}
