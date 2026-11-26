package com.reomusic.pa600v3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.ByteArrayOutputStream;

/** MIDI 1.0 streaming parser. Real-time bytes may occur anywhere, including SysEx. */
public final class NativeMidiLab {
    private int running=0,status=0,needed=0,count=0;
    private final byte[] data=new byte[2];
    private ByteArrayOutputStream sysex;
    private boolean discardSysEx=false;
    private static final int MAX_SYSEX=65536;
    public void reset(){running=0;status=0;needed=0;count=0;sysex=null;discardSysEx=false;}
    private static int length(int st){
        if(st>=0x80&&st<=0xef)return (st&0xf0)==0xc0||(st&0xf0)==0xd0?1:2;
        if(st==0xf1||st==0xf3)return 1;
        if(st==0xf2)return 2;
        return 0;
    }
    public List<byte[]> feed(byte[] bytes){
        ArrayList<byte[]> out=new ArrayList<>();
        for(byte raw:bytes){
            int v=raw&255;
            if(v>=0xf8){out.add(new byte[]{raw});continue;}
            if(discardSysEx){
                if(v==0xf7){discardSysEx=false;continue;}
                if(v<0x80)continue;
                discardSysEx=false;
            }
            if(sysex!=null){
                if(v==0xf7){sysex.write(v);out.add(sysex.toByteArray());sysex=null;continue;}
                if(v<0x80){if(sysex.size()<MAX_SYSEX-1)sysex.write(v);else{sysex=null;discardSysEx=true;}continue;}
                sysex=null; // interrupted, never emit partial SysEx
            }
            if(v>=0x80){
                count=0;status=0;needed=0;
                if(v==0xf0){running=0;sysex=new ByteArrayOutputStream();sysex.write(v);continue;}
                if(v>=0xf0)running=0;else running=v;
                if(v==0xf4||v==0xf5||v==0xf7)continue;
                int n=length(v);
                if(n==0){out.add(new byte[]{raw});continue;}
                status=v;needed=n;continue;
            }
            if(status==0){
                if(running==0)continue;
                status=running;needed=length(running);count=0;
            }
            data[count++]=raw;
            if(count==needed){
                byte[] msg=new byte[needed+1];msg[0]=(byte)status;
                System.arraycopy(data,0,msg,1,needed);out.add(msg);
                count=0;status=0;needed=0;
            }
        }
        return out;
    }
    private static byte[] b(int...x){byte[] r=new byte[x.length];for(int i=0;i<x.length;i++)r[i]=(byte)x[i];return r;}
    private static void assertMessages(List<byte[]> got,byte[]...want){
        if(got.size()!=want.length)throw new AssertionError("count "+got.size()+" vs "+want.length);
        for(int i=0;i<want.length;i++)if(!Arrays.equals(got.get(i),want[i]))throw new AssertionError("message "+i);
    }
    public static void main(String[] args){
        NativeMidiLab p=new NativeMidiLab();
        assertMessages(p.feed(b(0xcf,0x54)),b(0xcf,0x54));
        assertMessages(p.feed(b(0x55,0x56)),b(0xcf,0x55),b(0xcf,0x56));
        p.reset();assertMessages(p.feed(b(0xbf,99)));assertMessages(p.feed(b(2,98,64)),b(0xbf,99,2),b(0xbf,98,64));
        p.reset();assertMessages(p.feed(b(0xf0,0x42,0x7f,0xf8)),b(0xf8));assertMessages(p.feed(b(0x60,0x01,0xf7)),b(0xf0,0x42,0x7f,0x60,1,0xf7));
        p.reset();assertMessages(p.feed(b(0x90,60,0xfe,100,61,0)),b(0xfe),b(0x90,60,100),b(0x90,61,0));
        p.reset();assertMessages(p.feed(b(0xf0,1,0xc0,12)),b(0xc0,12));
        p.reset();assertMessages(p.feed(b(0xf2,1,2,3)),b(0xf2,1,2));
        p.reset();assertMessages(p.feed(b(0x90,60,100,0xf6,61,100)),b(0x90,60,100),b(0xf6));
        p.reset();assertMessages(p.feed(b(0xf0,1,0xf7,0xf0,2,0xf7)),b(0xf0,1,0xf7),b(0xf0,2,0xf7));
        System.out.println("NATIVE MIDI LAB: 9 streaming scenarios PASS");
    }
}
