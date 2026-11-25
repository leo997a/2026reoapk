package com.reomusic.pa600v3;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.content.Context;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.SharedPreferences;
import android.media.midi.*;
import android.media.AudioAttributes;
import android.media.SoundPool;
import android.content.Intent;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.text.InputType;
import android.view.*;
import android.widget.*;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.IOException;
import java.util.*;

public class MainActivity extends Activity {
    private static final int BG=0xff0c1724, PANEL=0xff182839, BUTTON=0xff294254,
       TEAL=0xff267d74, GOLD=0xff8a7142, WHITE=0xffeef5f4, MUTED=0xffa6bfcb, WARN=0xff80653a;
    private MidiManager midiManager;
    private MidiDevice midiDevice;
    private MidiInputPort midiSendPort;
    private MidiOutputPort midiReceivePort;
    private MidiReceiver monitorReceiver;
    private MidiManager.DeviceCallback deviceCallback;
    private final Handler handler=new Handler(Looper.getMainLooper());
    private SharedPreferences prefs;
    private LinearLayout frame, content, header, nav;
    private TextView connectionText, logText;
    private Spinner deviceSpinner;
    private final ArrayList<MidiDeviceInfo> deviceList=new ArrayList<>();
    private final ArrayList<String> eventLog=new ArrayList<>();
    private int tab=0, selectedBank=15, stylePage=0;
    private EditText styleName, stylePc, bookNumber, bookName, soundMsb, soundLsb, soundPc, soundCh;
    private TextView tapReadout;
    private final ArrayList<Long> taps=new ArrayList<>();
    private SoundPool padPool;
    private final int[] padSample=new int[4], padStream=new int[4];
    private final boolean[] padReady=new boolean[4];
    private final String[] padTitles={"DHOL","STABS","RIQ","FILL"};
    private final Button[] livePads=new Button[4];
    private TextView liveUsb;
    private int pendingImport=-1;
    private int padBpm=81;
    private boolean openingMidi=false;
    private int midiGeneration=0;
    private long lastClockNs=0;
    private final ArrayList<Long> clockIntervals=new ArrayList<>();
    private double measuredClockBpm=0;

    private int dp(float n){return (int)(getResources().getDisplayMetrics().density*n+.5f);}
    private GradientDrawable background(int color, int radius){
        GradientDrawable d=new GradientDrawable(); d.setColor(color);d.setCornerRadius(dp(radius));return d;
    }
    private LinearLayout vertical(){LinearLayout x=new LinearLayout(this);x.setOrientation(LinearLayout.VERTICAL);return x;}
    private TextView text(String s,int size,int color){
        TextView t=new TextView(this);t.setText(s);t.setTextColor(color);t.setTextSize(size);
        t.setGravity(Gravity.CENTER_VERTICAL);t.setTextDirection(View.TEXT_DIRECTION_FIRST_STRONG);return t;
    }
    private void gap(LinearLayout v,int h){View s=new View(this);v.addView(s,new LinearLayout.LayoutParams(1,dp(h)));}
    private void info(LinearLayout v,String s){TextView t=text(s,13,MUTED);v.addView(t);gap(v,8);}
    private Button button(String caption,int color,Runnable run){
        Button b=new Button(this);b.setText(caption);b.setAllCaps(false);b.setTextColor(WHITE);
        b.setTextSize(14);b.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        b.setPadding(dp(4),dp(5),dp(4),dp(5));b.setMinHeight(dp(58));b.setBackground(background(color,12));
        b.setOnClickListener(v->run.run());return b;
    }
    private void fullButton(LinearLayout root,String caption,int color,Runnable run){
        Button b=button(caption,color,run);LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,dp(58));
        lp.bottomMargin=dp(7);root.addView(b,lp);
    }
    private LinearLayout card(String title,String subtitle){
        LinearLayout c=vertical();c.setPadding(dp(12),dp(13),dp(12),dp(13));c.setBackground(background(PANEL,17));
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2);lp.bottomMargin=dp(12);
        content.addView(c,lp);TextView h=text(title,18,WHITE);h.setTypeface(Typeface.DEFAULT,Typeface.BOLD);c.addView(h);
        if(subtitle!=null&&!subtitle.isEmpty())info(c,subtitle);else gap(c,8);
        return c;
    }
    private void buttons(LinearLayout parent,String[] names,int[] pcs,int color){
        LinearLayout row=new LinearLayout(this);row.setOrientation(LinearLayout.HORIZONTAL);
        for(int i=0;i<names.length;i++){
            final int value=pcs[i];
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(64),1f);
            lp.setMargins(dp(3),dp(3),dp(3),dp(3));
            row.addView(button(names[i],color,()->pc(value)),lp);
        }
        parent.addView(row);
    }
    private EditText field(LinearLayout root,String name,String val,boolean numeric){
        root.addView(text(name,13,MUTED));
        EditText e=new EditText(this);e.setText(val);e.setTextColor(WHITE);e.setTextSize(17);
        e.setSingleLine(true);e.setSelectAllOnFocus(true);
        if(numeric)e.setInputType(InputType.TYPE_CLASS_NUMBER);
        root.addView(e,new LinearLayout.LayoutParams(-1,dp(53)));gap(root,5);return e;
    }
    private int num(EditText f,int min,int max){
        int n=Integer.parseInt(f.getText().toString().trim());
        if(n<min||n>max)throw new IllegalArgumentException("القيمة بين "+min+" و "+max);
        return n;
    }
    @Override public void onCreate(Bundle b){
        super.onCreate(b);
        getWindow().setStatusBarColor(BG);getWindow().setNavigationBarColor(BG);
        prefs=getSharedPreferences("reo-pa600-v3",MODE_PRIVATE);
        padBpm=prefs.getInt("virtual_pad_bpm",81);
        initPadAudio();
        getWindow().getDecorView().setSystemUiVisibility(5894);
        midiManager=(MidiManager)getSystemService(Context.MIDI_SERVICE);
        LinearLayout outer=vertical();outer.setLayoutDirection(View.LAYOUT_DIRECTION_LTR);outer.setBackgroundColor(BG);
        outer.setPadding(dp(10),dp(8),dp(10),dp(8));setContentView(outer);
        header=vertical();header.setPadding(dp(8),dp(6),dp(8),dp(5));outer.addView(header);
        TextView brand=text("REO MUSIC  /  Pa600QT",22,WHITE);brand.setTypeface(Typeface.DEFAULT,Typeface.BOLD);header.addView(brand);
        connectionText=text("USB MIDI · غير متصل",13,MUTED);header.addView(connectionText);gap(header,6);
        LinearLayout deviceRow=new LinearLayout(this);deviceRow.setOrientation(LinearLayout.HORIZONTAL);
        deviceSpinner=new Spinner(this);deviceSpinner.setBackground(background(PANEL,9));
        deviceRow.addView(deviceSpinner,new LinearLayout.LayoutParams(0,dp(47),1f));
        LinearLayout.LayoutParams refreshLp=new LinearLayout.LayoutParams(dp(67),dp(47));
        refreshLp.leftMargin=dp(6);deviceRow.addView(button("تحديث",BUTTON,()->refreshDevices()),refreshLp);
        LinearLayout.LayoutParams connectLp=new LinearLayout.LayoutParams(dp(67),dp(47));
        connectLp.leftMargin=dp(6);deviceRow.addView(button("اتصال",TEAL,()->connect()),connectLp);
        header.addView(deviceRow);
        HorizontalScrollView navScroll=new HorizontalScrollView(this);navScroll.setHorizontalScrollBarEnabled(false);
        nav=new LinearLayout(this);nav.setOrientation(LinearLayout.HORIZONTAL);navScroll.addView(nav);
        LinearLayout.LayoutParams navLp=new LinearLayout.LayoutParams(-1,dp(62));navLp.topMargin=dp(8);
        outer.addView(navScroll,navLp);
        ScrollView scroller=new ScrollView(this);scroller.setFillViewport(true);scroller.setVerticalScrollBarEnabled(false);
        content=vertical();content.setPadding(dp(4),dp(8),dp(4),dp(24));scroller.addView(content);
        outer.addView(scroller,new LinearLayout.LayoutParams(-1,0,1));
        deviceCallback=new MidiManager.DeviceCallback(){
            @Override public void onDeviceAdded(MidiDeviceInfo info){handler.post(()->refreshDevices());}
            @Override public void onDeviceRemoved(MidiDeviceInfo info){handler.post(()->{
                if(midiDevice!=null&&midiDevice.getInfo().getId()==info.getId()){closeMidi();connectionText.setText("USB MIDI · فُصل الجهاز");}
                refreshDevices();
            });}
        };
        if(midiManager!=null)midiManager.registerDeviceCallback(deviceCallback,handler);
        refreshDevices();show(0);handler.postDelayed(()->{if(midiSendPort==null&&!openingMidi)connect();},500);
    }
    private void refreshDevices(){
        if(deviceSpinner==null||midiManager==null)return;
        deviceList.clear();ArrayList<String> labels=new ArrayList<>();int selected=0;
        for(MidiDeviceInfo dev:midiManager.getDevices()){
            if(dev.getInputPortCount()<1)continue;
            deviceList.add(dev);
            String name=dev.getProperties().getString(MidiDeviceInfo.PROPERTY_NAME);
            if(name==null)name="USB MIDI "+dev.getId();
            labels.add(name);
            if(name.toLowerCase(Locale.ROOT).contains("pa600")||name.toLowerCase(Locale.ROOT).contains("korg"))selected=deviceList.size()-1;
        }
        if(labels.isEmpty())labels.add("لم يظهر جهاز USB MIDI");
        ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,labels);
        deviceSpinner.setAdapter(adapter);deviceSpinner.setSelection(selected);
    }
    private void closeMidi(){
        midiGeneration++;openingMidi=false;
        try{if(midiReceivePort!=null)midiReceivePort.close();}catch(Exception ignored){}
        try{if(midiSendPort!=null)midiSendPort.close();}catch(Exception ignored){}
        try{if(midiDevice!=null)midiDevice.close();}catch(Exception ignored){}
        midiReceivePort=null;midiSendPort=null;midiDevice=null;updateLiveUsb();
    }
    private void connect(){
        if(midiManager==null){connectionText.setText("نظام الهاتف لا يدعم MIDI");return;}
        int index=deviceSpinner.getSelectedItemPosition();
        if(index<0||index>=deviceList.size()){connectionText.setText("لا يوجد Pa600 متصل؛ تحقق من OTG");return;}
        if(openingMidi)return;
        MidiDeviceInfo selected=deviceList.get(index);
        closeMidi();openingMidi=true;final int token=++midiGeneration;connectionText.setText("جاري فتح MIDI...");updateLiveUsb();
        midiManager.openDevice(selected,d->{
            if(token!=midiGeneration){if(d!=null)try{d.close();}catch(Exception ignored){}return;}
            openingMidi=false;
            if(d==null){connectionText.setText("فشل فتح جهاز MIDI");updateLiveUsb();return;}
            MidiInputPort inp=d.openInputPort(0);
            if(inp==null){try{d.close();}catch(Exception ignored){}connectionText.setText("فشل فتح Input Port 0");updateLiveUsb();return;}
            midiDevice=d;midiSendPort=inp;
            if(d.getInfo().getOutputPortCount()>0){
                try{
                    monitorReceiver=new MidiReceiver(){
                        @Override public void onSend(byte[] data,int offset,int count,long timestamp){
                            byte[] packet=Arrays.copyOfRange(data,offset,offset+count);
                            handler.post(()->monitorIncoming(packet));
                        }
                    };
                    midiReceivePort=d.openOutputPort(0);
                    if(midiReceivePort!=null)midiReceivePort.connect(monitorReceiver);
                }catch(Exception ex){record("IN monitor not available");}
            }
            String name=selected.getProperties().getString(MidiDeviceInfo.PROPERTY_NAME);
            connectionText.setText("● متصل: "+(name==null?"MIDI":name)+" · OUT → USB");
            record("CONNECTED MIDI PORT 0");updateLiveUsb();
        },handler);
    }
    private String hex(byte[] data){
        StringBuilder b=new StringBuilder();
        for(byte x:data){if(b.length()>0)b.append(' ');b.append(String.format(Locale.ROOT,"%02X",x&255));}
        return b.toString();
    }
    private void record(String e){
        eventLog.add(0,e);
        while(eventLog.size()>32)eventLog.remove(eventLog.size()-1);
        if(logText!=null){StringBuilder sb=new StringBuilder();for(int i=0;i<Math.min(16,eventLog.size());i++)sb.append(eventLog.get(i)).append("\n");logText.setText(sb.toString());}
    }
    private void send(byte[] msg,String label){
        if(midiSendPort==null){connectionText.setText("غير متصل · اختر Pa600 واضغط اتصال");return;}
        try{midiSendPort.send(msg,0,msg.length);record("OUT → "+hex(msg)+" · "+label);}
        catch(IOException ex){connectionText.setText("خطأ USB MIDI: "+ex.getMessage());}
    }
    private void cc(int ch,int ctrl,int val){
        send(new byte[]{(byte)(0xB0|((ch-1)&15)),(byte)ctrl,(byte)val},"CC"+ctrl+" ch"+ch);
    }
    private void program(int ch,int pc){send(new byte[]{(byte)(0xC0|((ch-1)&15)),(byte)pc},"PC"+pc+" ch"+ch);}
    private void pc(int p){program(16,p);}
    private void sendStyle(int msb,int lsb,int p){
        cc(16,0,msb);cc(16,32,lsb);pc(p);
    }
    private void show(int index){
        tab=index;liveUsb=null;Arrays.fill(livePads,null);header.setVisibility(index==0?View.GONE:View.VISIBLE);nav.removeAllViews();
        String[] names={"العزف","الإيقاعات","SongBook","الأصوات","PAD ×4","الفحص"};
        for(int i=0;i<names.length;i++){
            final int idx=i;
            Button b=button(names[i],i==index?TEAL:PANEL,()->show(idx));
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(dp(i==2?110:91),dp(48));
            lp.setMargins(dp(3),dp(4),dp(3),dp(4));nav.addView(b,lp);
        }
        content.removeAllViews();logText=null;
        switch(index){
            case 0:renderLive();break;
            case 1:renderStyles();break;
            case 2:renderBook();break;
            case 3:renderSounds();break;
            case 4:renderPads();break;
            default:renderDiagnostic();break;
        }
    }
    private Button liveTile(String caption,int pcValue,int color){
        Button b=button(caption,color,()->pc(pcValue));
        b.setOnClickListener(v->{pc(pcValue);bHaptic(v);});
        b.setTextSize(12);
        b.setMinHeight(dp(51));
        b.setMinimumHeight(dp(51));
        b.setPadding(dp(2),dp(3),dp(2),dp(3));
        return b;
    }
    private void bHaptic(View v){v.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP);}
    private LinearLayout strip(LinearLayout root,String section,float portion){
        LinearLayout c=vertical();
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,-2,portion);
        lp.setMargins(dp(2),dp(2),dp(2),dp(2));
        c.setPadding(dp(5),dp(5),dp(5),dp(5));c.setBackground(background(PANEL,9));
        root.addView(c,lp);
        TextView t=text(section,11,MUTED);t.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        t.setGravity(Gravity.CENTER);c.addView(t,new LinearLayout.LayoutParams(-1,dp(19)));
        return c;
    }
    private void tiles(LinearLayout panel,String[] captions,int[] codes,int color){
        LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
        for(int i=0;i<captions.length;i++){
            Button b=liveTile(captions[i],codes[i],color);
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(51),1);
            lp.setMargins(dp(1),0,dp(1),0);r.addView(b,lp);
        }
        panel.addView(r);
    }
    private void pads(LinearLayout panel){
        LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
        for(int i=0;i<4;i++){
            final int idx=i;
            Button b=button("PAD "+(i+1)+"\n"+padTitles[i],padStream[i]!=0?TEAL:GOLD,()->toggleVirtualPad(idx));
            livePads[i]=b;
            b.setTextSize(11);b.setPadding(0,0,0,0);
            b.setOnLongClickListener(v->{configureVirtualPad(idx);return true;});
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(51),1);
            lp.setMargins(dp(1),0,dp(1),0);r.addView(b,lp);
        }
        panel.addView(r);
    }
    private void liveRow(LinearLayout wrapper){
        LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
        wrapper.addView(r,new LinearLayout.LayoutParams(-1,-2));
        LinearLayout pd=strip(r,"VIRTUAL PAD • AUDIO",4.1f);pads(pd);
        LinearLayout sts=strip(r,"STS",4.1f);
        tiles(sts,new String[]{"1","2","3","4"},new int[]{64,65,66,67},BUTTON);
        LinearLayout extra=strip(r,"LIVE CONTROL",3.5f);
        tiles(extra,new String[]{"MEM","BASS","AUTO"},new int[]{98,100,97},BUTTON);
    }
    private void elementRow(LinearLayout wrapper){
        LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
        wrapper.addView(r,new LinearLayout.LayoutParams(-1,-2));
        LinearLayout intro=strip(r,"INTRO",3.1f);
        tiles(intro,new String[]{"1","2","3"},new int[]{80,81,82},BUTTON);
        LinearLayout variation=strip(r,"VARIATIONS",4.3f);
        tiles(variation,new String[]{"1","2","3","4"},new int[]{83,84,85,86},TEAL);
        LinearLayout breakBox=strip(r,"BREAK",1.1f);
        tiles(breakBox,new String[]{"●"},new int[]{91},GOLD);
        LinearLayout ending=strip(r,"ENDING",3.1f);
        tiles(ending,new String[]{"1","2","3"},new int[]{92,93,94},BUTTON);
    }
    private void lowerRow(LinearLayout wrapper){
        LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
        wrapper.addView(r,new LinearLayout.LayoutParams(-1,-2));
        LinearLayout fills=strip(r,"DIRECT FILL (EXTRA)",4.1f);
        tiles(fills,new String[]{"1","2","3","4"},new int[]{87,88,89,90},BUTTON);
        LinearLayout play=strip(r,"STYLE TRANSPORT",2.5f);
        tiles(play,new String[]{"▶ / ■"},new int[]{103},TEAL);
        LinearLayout secondary=strip(r,"EXTRA CONTROL",5f);
        tiles(secondary,new String[]{"FADE","STS MD","B. INV","T.LOCK"},new int[]{95,96,99,101},BUTTON);
    }
    private void renderLive(){
        // Stage-performance layout: fixed hardware-inspired positions, no menus or scrolling required for essentials.
        LinearLayout top=new LinearLayout(this);top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout.LayoutParams tp=new LinearLayout.LayoutParams(-1,dp(35));content.addView(top,tp);
        TextView title=text("REO  |  Pa600QT     •     LIVE CONSOLE",16,WHITE);
        title.setTypeface(Typeface.DEFAULT,Typeface.BOLD);
        top.addView(title,new LinearLayout.LayoutParams(0,-1,1));
        liveUsb=text("",12,MUTED);top.addView(liveUsb);updateLiveUsb();
        Button reconnect=button("USB",BUTTON,()->{refreshDevices();connect();});
        reconnect.setTextSize(10);
        LinearLayout.LayoutParams usb=new LinearLayout.LayoutParams(dp(55),dp(35));
        usb.leftMargin=dp(9);top.addView(reconnect,usb);
        LinearLayout console=vertical();
        console.setPadding(dp(2),dp(2),dp(2),dp(2));
        content.addView(console,new LinearLayout.LayoutParams(-1,-2));
        liveRow(console);
        elementRow(console);
        lowerRow(console);
        TextView hint=text("VIRTUAL PAD: صوت من الهاتف · الضغط مطولاً لاختيار WAV · BPM "+padBpm+" · لا يضغط PAD الداخلي",11,MUTED);
        hint.setPadding(dp(5),dp(3),dp(5),0);
        content.addView(hint,new LinearLayout.LayoutParams(-1,-2));
        TextView next=text("بنوك Styles / SongBook / الأصوات / فحص MIDI في التبويبات أعلاه",11,MUTED);
        content.addView(next);
        fullButton(content,"STOP ALL VIRTUAL PADS",BUTTON,()->stopAllPads());
    }
    private void tapTempo(){
        long now=SystemClock.uptimeMillis();
        if(!taps.isEmpty()&&now-taps.get(taps.size()-1)>2500)taps.clear();
        taps.add(now);if(taps.size()>6)taps.remove(0);
        if(taps.size()>=2){
            long span=taps.get(taps.size()-1)-taps.get(0);
            int bpm=(int)Math.round(60000.0*(taps.size()-1)/Math.max(1,span));
            tapReadout.setText(bpm+" BPM · مرجع");
        }
    }
    private String[] bankLabels(){
        String[] names=new String[15];for(int i=0;i<12;i++)names[i]="Favorite "+(i+1);
        for(int i=0;i<3;i++)names[12+i]="User "+(i+1);return names;
    }
    private void renderStyles(){
        LinearLayout c=card("مكتبة الإيقاعات","بنوك Korg الموثقة: Favorite 1–12 / User 1–3؛ قد تختلف أسماء إيقاعات SET");
        Spinner bank=new Spinner(this);
        ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,bankLabels());
        bank.setAdapter(adapter);bank.setSelection(Math.max(0,Math.min(14,selectedBank-15)));
        bank.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener(){
            @Override public void onItemSelected(android.widget.AdapterView<?> parent,View v,int i,long id){selectedBank=15+i;}
            @Override public void onNothingSelected(android.widget.AdapterView<?> p){}
        });
        c.addView(bank,new LinearLayout.LayoutParams(-1,dp(54)));
        final LinearLayout grid=vertical();TextView pageText=text("",14,MUTED); 
        c.addView(pageText);
        c.addView(grid);
        Runnable repaint=()->{
            pageText.setText("الصفحة "+(stylePage+1)+" من 5 · المواقع "+(stylePage*8+1)+"–"+(stylePage*8+8));
            grid.removeAllViews();
            for(int row=0;row<2;row++){
                LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
                for(int col=0;col<4;col++){
                    final int pos=stylePage*8+row*4+col+1;
                    LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(62),1);lp.setMargins(dp(3),dp(3),dp(3),dp(3));
                    r.addView(button(String.valueOf(pos),BUTTON,()->sendStyle(0,selectedBank,pos)),lp);
                }
                grid.addView(r);
            }
        };
        LinearLayout pageBar=new LinearLayout(this);pageBar.setOrientation(LinearLayout.HORIZONTAL);
        for(int i=0;i<2;i++){
            final boolean back=i==0;
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(58),1);lp.setMargins(dp(3),dp(5),dp(3),dp(5));
            pageBar.addView(button(back?"◀ السابق":"التالي ▶",BUTTON,()->{stylePage=(stylePage+(back?4:1))%5;repaint.run();}),lp);
        }
        c.addView(pageBar);repaint.run();gap(c,12);
        info(c,"أرقام المواقع 1–40 تتبع جدول MIDI الرسمي، وقد تكون بعض المواقع فارغة.");
        stylePc=field(c,"Program Change مخصص (0–127)","1",true);
        styleName=field(c,"اسم المفضلة (مثال: شيخاني)","",false);
        fullButton(c,"استدعاء رقم البرنامج المخصص",TEAL,()->{
            try{sendStyle(0,selectedBank,num(stylePc,0,127));}catch(Exception e){message(e.getMessage());}
        });
        fullButton(c,"حفظ الإيقاع في المفضلة",GOLD,()->saveStyleFavorite());
        c=card("إيقاعاتك المفضلة","اضغط للاستدعاء، واضغط مطولاً للحذف");
        addFavoriteButtons(c,"styles");
        c=card("مكتبة بنوك مخصصة","للبنوك الأخرى MSB + LSB + PC (كل قيمة 0–127)");
        final EditText msb=field(c,"Bank MSB","0",true);
        final EditText lsb=field(c,"Bank LSB","15",true);
        final EditText prog=field(c,"Program","1",true);
        fullButton(c,"إرسال Bank Select + Program",BUTTON,()->{
            try{sendStyle(num(msb,0,127),num(lsb,0,127),num(prog,0,127));}catch(Exception ex){message(ex.getMessage());}
        });
    }
    private void saveStyleFavorite(){
        try{
            int p=num(stylePc,0,127);
            String name=styleName.getText().toString().trim();
            if(name.isEmpty())name="Style "+(selectedBank-14)+" / "+p;
            JSONObject item=new JSONObject();item.put("name",name);item.put("type","styles");
            item.put("msb",0);item.put("lsb",selectedBank);item.put("program",p);
            appendFavorite("styles",item);message("حُفظ "+name);show(1);
        }catch(Exception ex){message(ex.getMessage());}
    }
    private void renderBook(){
        LinearLayout c=card("SongBook بالأرقام","الاستدعاء الرسمي NRPN 99/98 ثم Data Entry 6/38 على قناة Control 16");
        info(c,"اختر رقم Entry الموجود داخل SongBook؛ رقم الأغنية ليس رقم ملف MIDI.");
        bookNumber=field(c,"رقم SongBook (0–9999)","77",true);
        bookName=field(c,"اسم للأغنية في الهاتف","",false);
        fullButton(c,"استدعاء من SongBook",TEAL,()->{
            try{songBook(num(bookNumber,0,9999));}catch(Exception ex){message(ex.getMessage());}
        });
        fullButton(c,"حفظ SongBook بالمفضلة",GOLD,()->{
            try{
                int n=num(bookNumber,0,9999);String name=bookName.getText().toString().trim();
                if(name.isEmpty())name="SongBook #"+n;
                JSONObject item=new JSONObject();item.put("type","songbook");item.put("name",name);item.put("number",n);
                appendFavorite("songbook",item);show(2);
            }catch(Exception ex){message(ex.getMessage());}
        });
        c=card("قائمة أغانيك","اضغط للاستدعاء، اضغط مطولاً للحذف");
        addFavoriteButtons(c,"songbook");
        c=card("ما الذي يفعله SongBook؟","يمكن أن يستدعي الإيقاع والـSTS وإعدادات الأغنية التي خزنتها على الأورك. الاستدعاء لا يبدأ الإيقاع تلقائياً.");
        fullButton(c,"START / STOP",BUTTON,()->pc(103));
    }
    private void songBook(int n){
        cc(16,99,2);cc(16,98,64);cc(16,6,n/100);cc(16,38,n%100);
        record("SONGBOOK #"+n+" · NRPN");
    }
    private void renderSounds(){
        LinearLayout c=card("الأصوات عبر MIDI","مهم: إعداد Tablet قد يضع Ch01=Off. قبل استخدام الأصوات اجعل قناة مناسبة في GLOBAL → MIDI IN = Upper 1.");
        soundCh=field(c,"قناة الصوت (1–15)، لا تستخدم 16 للتحكم بالصوت","1",true);
        soundMsb=field(c,"MSB","121",true);
        soundLsb=field(c,"LSB","10",true);
        soundPc=field(c,"Program","0",true);
        fullButton(c,"اختيار الصوت",TEAL,()->{
            try{sendSound(num(soundCh,1,15),num(soundMsb,0,127),num(soundLsb,0,127),num(soundPc,0,127));}
            catch(Exception ex){message(ex.getMessage());}
        });
        fullButton(c,"Grand Piano RX · صوت مصنع موثق",BUTTON,()->{
            try{sendSound(num(soundCh,1,15),121,10,0);}catch(Exception ex){message(ex.getMessage());}
        });
        c=card("قوة الصوت والتعبير","تؤثر على مسار MIDI المعيّن للقناة المختارة، وليس بالضرورة Master Volume.");
        c.addView(text("Volume CC7",14,WHITE));SeekBar vol=new SeekBar(this);vol.setMax(127);vol.setProgress(100);c.addView(vol);
        vol.setOnSeekBarChangeListener(seek(7));
        c.addView(text("Expression CC11",14,WHITE));SeekBar expr=new SeekBar(this);expr.setMax(127);expr.setProgress(100);c.addView(expr);
        expr.setOnSeekBarChangeListener(seek(11));
        fullButton(c,"اختبار نوتة C4 (قناة الصوت المختارة)",BUTTON,()->{
            try{note(num(soundCh,1,15),60,98,350);}catch(Exception ex){message(ex.getMessage());}
        });
    }
    private SeekBar.OnSeekBarChangeListener seek(int ccNum){
        return new SeekBar.OnSeekBarChangeListener(){
            @Override public void onProgressChanged(SeekBar s,int value,boolean fromUser){
                if(fromUser)try{cc(num(soundCh,1,15),ccNum,value);}catch(Exception ex){message("اختر قناة صالحة");}
            }
            @Override public void onStartTrackingTouch(SeekBar s){}
            @Override public void onStopTrackingTouch(SeekBar s){}
        };
    }
    private void sendSound(int ch,int msb,int lsb,int p){
        cc(ch,0,msb);cc(ch,32,lsb);program(ch,p);
    }
    private void note(int ch,int number,int velocity,long ms){
        send(new byte[]{(byte)(0x90|(ch-1)),(byte)number,(byte)velocity},"NOTE ON ch"+ch);
        handler.postDelayed(()->send(new byte[]{(byte)(0x80|(ch-1)),(byte)number,0},"NOTE OFF ch"+ch),ms);
    }
    private void updateLiveUsb(){
        if(liveUsb==null)return;
        liveUsb.setText(openingMidi?"● USB OPENING":midiSendPort!=null?"● USB CONNECTED":"● USB DISCONNECTED");
        liveUsb.setTextColor(midiSendPort!=null?0xff80d9ad:0xffe9ba78);
    }
    private void monitorIncoming(byte[] p){
        boolean other=false;
        for(byte x:p){
            int n=x&255;
            if(n==0xf8){
                long now=System.nanoTime();
                if(lastClockNs>0){long d=now-lastClockNs;if(d>5000000L&&d<150000000L){
                    clockIntervals.add(d);if(clockIntervals.size()>48)clockIntervals.remove(0);
                    if(clockIntervals.size()>=24){long sum=0;for(long t:clockIntervals)sum+=t;measuredClockBpm=60000000000.0/(24.0*sum/clockIntervals.size());}
                }}
                lastClockNs=now;
            }else if(n!=0xfe)other=true;
        }
        if(other)record("IN ← "+hex(p));
    }
    private void initPadAudio(){
        padPool=new SoundPool.Builder().setMaxStreams(4).setAudioAttributes(new AudioAttributes.Builder()
          .setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build()).build();
        padPool.setOnLoadCompleteListener((pool,id,status)->{
            for(int i=0;i<4;i++)if(padSample[i]==id){padReady[i]=status==0;if(status!=0)record("PAD "+(i+1)+" audio load failed");}
        });
        for(int i=0;i<4;i++)loadPad(i);
    }
    private void loadPad(int i){
        stopVirtualPad(i);padReady[i]=false;
        if(padSample[i]!=0)padPool.unload(padSample[i]);
        padSample[i]=0;
        try{
            java.io.File custom=new java.io.File(getFilesDir(),"pad"+i+".wav");
            if(custom.exists())padSample[i]=padPool.load(custom.getAbsolutePath(),1);
            else try(android.content.res.AssetFileDescriptor fd=getAssets().openFd("pad"+(i+1)+".wav")){
                padSample[i]=padPool.load(fd,1);
            }
        }catch(Exception ex){record("PAD "+(i+1)+" load: "+ex.getMessage());}
    }
    private void updatePadTile(int i){
        Button b=livePads[i];if(b==null)return;
        b.setBackground(background(padStream[i]!=0?TEAL:GOLD,12));
        b.setText("PAD "+(i+1)+"\n"+(padStream[i]!=0?"● PLAY":padTitles[i]));
    }
    private void stopVirtualPad(int i){
        if(padPool!=null&&padStream[i]!=0)padPool.stop(padStream[i]);
        padStream[i]=0;updatePadTile(i);
    }
    private void stopAllPads(){for(int i=0;i<4;i++)stopVirtualPad(i);}
    private void toggleVirtualPad(int i){
        if(padStream[i]!=0){stopVirtualPad(i);return;}
        if(!padReady[i]){message("صوت PAD غير جاهز؛ جرّب مجدداً");return;}
        float rate=(float)padBpm/81f;
        if(prefs.getBoolean("pad_custom"+i,false))rate=1f;
        boolean loop=prefs.getBoolean("pad_loop"+i,i!=3);
        padStream[i]=padPool.play(padSample[i],0.72f,0.72f,1,loop?-1:0,rate);
        updatePadTile(i);
        if(!loop){int stream=padStream[i];handler.postDelayed(()->{if(padStream[i]==stream){padStream[i]=0;updatePadTile(i);}},(long)(2*60000.0/padBpm)+80);}
        if(padStream[i]==0)message("تعذر تشغيل PAD");
    }
    private void configureVirtualPad(int i){
        String[] choices={"استيراد WAV قصير من الهاتف","تبديل Loop / One-shot","استعادة الصوت الأصلي"};
        new AlertDialog.Builder(this).setTitle("Virtual PAD "+(i+1)).setItems(choices,(d,which)->{
            if(which==0){
                pendingImport=i;Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);intent.setType("audio/*");
                startActivityForResult(intent,701);
            }else if(which==1){
                boolean loop=!prefs.getBoolean("pad_loop"+i,i!=3);
                prefs.edit().putBoolean("pad_loop"+i,loop).apply();stopVirtualPad(i);message(loop?"Loop":"One-shot");
            }else{
                new java.io.File(getFilesDir(),"pad"+i+".wav").delete();
                prefs.edit().remove("pad_custom"+i).apply();loadPad(i);
            }
        }).setNegativeButton("إلغاء",null).show();
    }
    @Override protected void onActivityResult(int request,int result,Intent data){
        super.onActivityResult(request,result,data);
        if(request!=701||result!=RESULT_OK||data==null||pendingImport<0)return;
        int i=pendingImport;pendingImport=-1;
        java.io.File tmp=new java.io.File(getFilesDir(),"pad"+i+".tmp");
        try(java.io.InputStream in=getContentResolver().openInputStream(data.getData());java.io.FileOutputStream out=new java.io.FileOutputStream(tmp)){
            byte[] buf=new byte[8192];int n,total=0;
            while((n=in.read(buf))>0){total+=n;if(total>480000)throw new IOException("WAV must be under 480 KB");out.write(buf,0,n);}
        }catch(Exception ex){tmp.delete();message("اختر WAV PCM قصيراً أقل من 480KB");return;}
        try(java.io.RandomAccessFile r=new java.io.RandomAccessFile(tmp,"r")){
            if(r.readInt()!=0x52494646){throw new IOException("RIFF");}r.seek(8);if(r.readInt()!=0x57415645)throw new IOException("WAVE");
        }catch(Exception ex){tmp.delete();message("الملف ليس WAV");return;}
        java.io.File dst=new java.io.File(getFilesDir(),"pad"+i+".wav");
        if(!tmp.renameTo(dst)){tmp.delete();message("تعذر حفظ الملف");return;}
        prefs.edit().putBoolean("pad_custom"+i,true).apply();loadPad(i);message("تم استيراد WAV إلى PAD "+(i+1));
    }
    private void renderPads(){
        LinearLayout c=card("VIRTUAL PADS · محرك صوت الهاتف","أربعة مقاطع أصلية قصيرة بسرعة 81 BPM. تعمل حتى دون USB. لا تمثل تشغيل PAD الداخلي للأورك.");
        info(c,"اضغط مرة للتشغيل ومرة للإيقاف. اخرج صوت الهاتف إلى الميكسر عبر مسار صوت مناسب. USB MIDI لا ينقل هذا الصوت.");
        for(int i=0;i<4;i++){
            final int k=i;fullButton(c,"PAD "+(i+1)+" · "+padTitles[i],GOLD,()->toggleVirtualPad(k));
            fullButton(c,"إعداد / استيراد WAV "+(i+1),BUTTON,()->configureVirtualPad(k));
        }
        fullButton(c,"STOP ALL PADS",TEAL,()->stopAllPads());
        c=card("BPM","تغيير سرعة المقاطع الأصلية يغيّر الطبقة أيضاً؛ لا توجد مزامنة طور تلقائية مع الأورك.");
        EditText bpm=field(c,"سرعة الهاتف 50–120",String.valueOf(padBpm),true);
        fullButton(c,"تطبيق السرعة",TEAL,()->{try{stopAllPads();padBpm=num(bpm,50,120);prefs.edit().putInt("virtual_pad_bpm",padBpm).apply();message("BPM "+padBpm);}catch(Exception ex){message(ex.getMessage());}});
        fullButton(c,"قراءة MIDI Clock (تقريبية)",BUTTON,()->message(measuredClockBpm>0&&System.nanoTime()-lastClockNs<1000000000L?String.format(Locale.ROOT,"%.1f BPM",measuredClockBpm):"لا توجد ساعة MIDI واردة حديثاً"));
        c=card("PAD الداخلي للأورك","لم نجد أمر تشغيل MIDI موثق. لا تغير Tablet أو قنوات Control. استخدم مراقب MIDI لفحص الأزرار المادية.");
        fullButton(c,"MIDI Monitor",BUTTON,()->show(5));
    }
    private void appendFavorite(String key,JSONObject item){
        JSONArray a;
        try{a=new JSONArray(prefs.getString(key,"[]"));}catch(Exception ex){a=new JSONArray();}
        a.put(item);prefs.edit().putString(key,a.toString()).apply();
    }
    private void addFavoriteButtons(LinearLayout c,String key){
        try{
            JSONArray a=new JSONArray(prefs.getString(key,"[]"));
            if(a.length()==0){info(c,"لا توجد عناصر محفوظة بعد");return;}
            for(int i=0;i<a.length();i++){
                JSONObject item=a.getJSONObject(i);final int idx=i;
                Button b=button(item.optString("name","Favorite"),BUTTON,()->{
                    if(key.equals("styles"))sendStyle(item.optInt("msb",0),item.optInt("lsb",15),item.optInt("program",1));
                    else songBook(item.optInt("number",0));
                });
                b.setOnLongClickListener(v->{
                    new AlertDialog.Builder(this).setTitle("حذف من المفضلة؟").setMessage(item.optString("name"))
                        .setPositiveButton("حذف",(d,w)->removeFavorite(key,idx))
                        .setNegativeButton("إلغاء",null).show();return true;
                });
                LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,dp(58));lp.bottomMargin=dp(7);c.addView(b,lp);
            }
        }catch(Exception ex){info(c,"خطأ في قراءة المفضلة");}
    }
    private void removeFavorite(String key,int index){
        try{
            JSONArray all=new JSONArray(prefs.getString(key,"[]"));JSONArray next=new JSONArray();
            for(int i=0;i<all.length();i++)if(i!=index)next.put(all.get(i));
            prefs.edit().putString(key,next.toString()).apply();show(tab);
        }catch(Exception ex){message("تعذر الحذف");}
    }
    private void renderDiagnostic(){
        LinearLayout c=card("USB / MIDI Diagnostics","اتصال الجهاز يشير إلى فتح المنفذ فقط. لا يمكن معرفة حالة LED الخاصة بالأزرار من رسائل الإرسال.");
        fullButton(c,"اختبار VAR 2 · CF 54",TEAL,()->pc(84));
        fullButton(c,"اختبار START / STOP · CF 67",BUTTON,()->pc(103));
        fullButton(c,"اختبار C4 ch1 · قد لا يعمل مع Tablet",BUTTON,()->note(1,60,90,300));
        c=card("MIDI Monitor","IN يعني بيانات خرجت من الأورك ووصلت الهاتف. OUT يعني التطبيق أرسلها. بعض أزرار PAD قد لا ترسل رسالة قابلة للرصد.");
        logText=text("",12,MUTED);logText.setTypeface(Typeface.MONOSPACE);
        c.addView(logText);record("MONITOR READY");
        c=card("نسخة المكتبة الاحتياطية","انسخ بيانات المفضلة والحفظ إلى الحافظة كي لا تضيع عند تغيير الإصدار.");
        fullButton(c,"نسخ مكتبة الإيقاعات والأغاني",TEAL,()->exportLibrary());
        fullButton(c,"استيراد مكتبة من الحافظة",BUTTON,()->importLibrary());
        c=card("المراجع والتنبيهات","Pa600QT OS 2.1: Control 16. Virtual PAD صوت محلي وليس تشغيل PAD الداخلي.");
        info(c,"الإصدار REO v4.1 · Galaxy A52s · يعمل محلياً بدون إنترنت بعد التثبيت.");
    }
    private void exportLibrary(){
        try{
            JSONObject all=new JSONObject();all.put("format","REO_PA600_V3");
            all.put("styles",new JSONArray(prefs.getString("styles","[]")));
            all.put("songbook",new JSONArray(prefs.getString("songbook","[]")));
            ClipboardManager cb=(ClipboardManager)getSystemService(CLIPBOARD_SERVICE);
            cb.setPrimaryClip(ClipData.newPlainText("REO Pa600 backup",all.toString()));
            message("نُسخت المكتبة إلى الحافظة. احتفظ بها في ملف نصي خاص.");
        }catch(Exception ex){message("فشل النسخ: "+ex.getMessage());}
    }
    private void importLibrary(){
        try{
            ClipboardManager cb=(ClipboardManager)getSystemService(CLIPBOARD_SERVICE);
            if(cb==null||!cb.hasPrimaryClip()){message("الحافظة فارغة");return;}
            CharSequence raw=cb.getPrimaryClip().getItemAt(0).coerceToText(this);
            if(raw==null||raw.length()>250000){message("محتوى الحافظة غير صالح أو كبير جداً");return;}
            JSONObject o=new JSONObject(raw.toString());
            if(!"REO_PA600_V3".equals(o.optString("format"))){message("صيغة مكتبة غير مدعومة");return;}
            JSONArray styles=o.getJSONArray("styles"),songs=o.getJSONArray("songbook");
            if(styles.length()>300||songs.length()>300)throw new IllegalArgumentException("تجاوز الحد");
            new AlertDialog.Builder(this).setTitle("استيراد المكتبة")
              .setMessage("استبدال المفضلة الحالية بالبيانات الموجودة في الحافظة؟")
              .setPositiveButton("استيراد",(d,w)->{
                prefs.edit().putString("styles",styles.toString()).putString("songbook",songs.toString()).apply();
                message("تم استيراد المكتبة");show(tab);
              }).setNegativeButton("إلغاء",null).show();
        }catch(Exception ex){message("خطأ في بيانات الحافظة: "+ex.getMessage());}
    }
    private void message(String msg){Toast.makeText(this,msg,Toast.LENGTH_LONG).show();}
    @Override protected void onDestroy(){
        if(midiManager!=null&&deviceCallback!=null)midiManager.unregisterDeviceCallback(deviceCallback);
        stopAllPads();if(padPool!=null){padPool.release();padPool=null;}closeMidi();handler.removeCallbacksAndMessages(null);super.onDestroy();
    }
}
