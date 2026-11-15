package com.reomusic.pa600;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.content.Context;
import android.content.SharedPreferences;
import android.text.InputType;
import org.json.JSONArray;
import org.json.JSONObject;
import android.media.midi.*;
import android.widget.*;
import android.view.*;
import java.util.*;
import java.io.IOException;

public class MainActivity extends Activity {
  private static final int BG=0xff0e1725, SURFACE=0xff1c2b39, TILE=0xff294454, GREEN=0xff32675c, TXT=0xfff0f4f2, SOFT=0xffb0c7cf;
  private final Handler handler=new Handler(Looper.getMainLooper());
  private MidiManager manager;
  private MidiDevice device;
  private MidiInputPort port;
  private MidiDeviceInfo[] choices=new MidiDeviceInfo[0];
  private Spinner spinner;
  private EditText styleMsb, styleLsb, stylePc, styleLabel;
  private LinearLayout favoritesList;
  private SharedPreferences prefs;
  private TextView status, log;
  private final List<String> messages=new ArrayList<>();
  private MidiManager.DeviceCallback listener;
  private int dp(int x){return (int)(getResources().getDisplayMetrics().density*x+.5f);}
  private GradientDrawable surface(int color){GradientDrawable d=new GradientDrawable();d.setColor(color);d.setCornerRadius(dp(13));return d;}
  private TextView label(String text,int size,int color){TextView v=new TextView(this);v.setText(text);v.setTextSize(size);v.setTextColor(color);v.setGravity(Gravity.CENTER_VERTICAL);return v;}
  private LinearLayout column(){LinearLayout l=new LinearLayout(this);l.setOrientation(LinearLayout.VERTICAL);return l;}
  private void space(LinearLayout p,int n){View v=new View(this);p.addView(v,new LinearLayout.LayoutParams(1,dp(n)));}
  private Button button(String text,int color,Runnable action){
    Button b=new Button(this);b.setAllCaps(false);b.setText(text);b.setTextSize(13);b.setTypeface(Typeface.DEFAULT,Typeface.BOLD);b.setTextColor(TXT);
    b.setBackground(surface(color));b.setMinHeight(dp(65));b.setPadding(dp(4),dp(5),dp(4),dp(5));
    b.setOnClickListener(v->action.run());return b;
  }
  private void row(LinearLayout parent,String[] names,int[] codes,int color){
    LinearLayout r=new LinearLayout(this);r.setOrientation(LinearLayout.HORIZONTAL);
    for(int i=0;i<names.length;i++){
      final int code=codes[i];
      LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(0,dp(66),1f);
      p.setMargins(dp(3),dp(3),dp(3),dp(3));r.addView(button(names[i],color,()->sendPc(code)),p);
    }
    parent.addView(r,new LinearLayout.LayoutParams(-1,-2));
  }
  private LinearLayout section(LinearLayout parent,String title){
    LinearLayout c=column();c.setPadding(dp(12),dp(12),dp(12),dp(12));c.setBackground(surface(SURFACE));
    LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,-2);p.bottomMargin=dp(12);parent.addView(c,p);
    c.addView(label(title,17,TXT),new LinearLayout.LayoutParams(-1,-2));space(c,8);return c;
  }
  @Override public void onCreate(Bundle saved){
    super.onCreate(saved);
    getWindow().setStatusBarColor(BG);getWindow().setNavigationBarColor(BG);
    manager=(MidiManager)getSystemService(Context.MIDI_SERVICE);
    prefs=getSharedPreferences("reo-pa600-a52s",MODE_PRIVATE);
    ScrollView scroll=new ScrollView(this);scroll.setFillViewport(true);scroll.setBackgroundColor(BG);
    LinearLayout root=column();root.setPadding(dp(12),dp(16),dp(12),dp(24));scroll.addView(root);
    setContentView(scroll);
    root.addView(label("REO MUSIC  /  Pa600 Remote",24,TXT));space(root,6);
    root.addView(label("Samsung Galaxy A52s · Android USB MIDI",12,SOFT));space(root,14);
    LinearLayout conn=section(root,"USB MIDI  ·  الاتصال");
    spinner=new Spinner(this);conn.addView(spinner,new LinearLayout.LayoutParams(-1,dp(54)));
    LinearLayout r=new LinearLayout(this);
    LinearLayout.LayoutParams half=new LinearLayout.LayoutParams(0,dp(62),1f);
    half.setMargins(dp(3),dp(2),dp(3),dp(2));
    r.addView(button("تحديث",TILE,()->refresh()),half);
    LinearLayout.LayoutParams half2=new LinearLayout.LayoutParams(0,dp(62),1f);
    half2.setMargins(dp(3),dp(2),dp(3),dp(2));
    r.addView(button("اتصال",GREEN,()->connect()),half2);conn.addView(r);
    status=label("اختر Pa600 ثم اضغط اتصال",14,SOFT);conn.addView(status);
    space(conn,6);conn.addView(label("قناة الإرسال 16  |  GLOBAL → MIDI IN → CH16 = Control",12,SOFT));
    LinearLayout play=section(root,"تشغيل الإيقاع");
    row(play,new String[]{"START / STOP","AUTO FILL","FADE"},new int[]{103,97,95},TILE);
    LinearLayout var=section(root,"Variations");
    row(var,new String[]{"VAR 1","VAR 2","VAR 3","VAR 4"},new int[]{83,84,85,86},GREEN);
    row(var,new String[]{"FILL 1","FILL 2","FILL 3","FILL 4"},new int[]{87,88,89,90},TILE);
    row(var,new String[]{"BREAK","MEMORY"},new int[]{91,98},TILE);
    LinearLayout arr=section(root,"Intro / Ending");
    row(arr,new String[]{"INTRO 1","INTRO 2","INTRO 3"},new int[]{80,81,82},TILE);
    row(arr,new String[]{"END 1","END 2","END 3"},new int[]{92,93,94},TILE);
    LinearLayout sts=section(root,"STS / Bass");
    row(sts,new String[]{"STS 1","STS 2","STS 3","STS 4"},new int[]{64,65,66,67},TILE);
    row(sts,new String[]{"BASS INV","MANUAL","TEMPO LOCK"},new int[]{99,100,101},TILE);
    LinearLayout styles=section(root,"اختيار الإيقاعات · STYLE BANK");
    styles.addView(label("أرقام البنك تختلف حسب الإيقاع. أدخل القيم من قائمة Styles أو دليل Korg.",12,SOFT));
    space(styles,8);
    styleLabel=new EditText(this);styleLabel.setSingleLine(true);styleLabel.setHint("اسم الإيقاع للمفضلة");styleLabel.setHintTextColor(SOFT);styleLabel.setTextColor(TXT);
    styles.addView(styleLabel,new LinearLayout.LayoutParams(-1,dp(52)));
    styleMsb=field(styles,"MSB · CC0",0);
    styleLsb=field(styles,"LSB · CC32",0);
    stylePc=field(styles,"Program · 0–127",0);
    LinearLayout actions=new LinearLayout(this);
    LinearLayout.LayoutParams actionLp=new LinearLayout.LayoutParams(0,dp(65),1);
    actionLp.setMargins(dp(3),dp(3),dp(3),dp(3));
    actions.addView(button("استدعاء الإيقاع",GREEN,()->sendStyle()),actionLp);
    LinearLayout.LayoutParams saveLp=new LinearLayout.LayoutParams(0,dp(65),1);
    saveLp.setMargins(dp(3),dp(3),dp(3),dp(3));
    actions.addView(button("حفظ بالمفضلة",TILE,()->saveFavorite()),saveLp);
    styles.addView(actions);
    favoritesList=column();styles.addView(favoritesList,new LinearLayout.LayoutParams(-1,-2));
    refreshFavorites();
    LinearLayout diag=section(root,"اختبار الإرسال");
    diag.addView(button("اختبار نوتة C4 على قناة 1",TILE,()->testNote()),new LinearLayout.LayoutParams(-1,dp(64)));
    space(diag,8);log=label("لا يوجد سجل بعد",13,SOFT);diag.addView(log);
    diag.addView(label("سجل الإرسال لا يثبت استجابة الأورك. جرّب الأزرار في STYLE PLAY.",12,SOFT));
    refresh();
    listener=new MidiManager.DeviceCallback(){
      @Override public void onDeviceAdded(MidiDeviceInfo info){handler.post(()->refresh());}
      @Override public void onDeviceRemoved(MidiDeviceInfo info){handler.post(()->{if(device!=null&&device.getInfo().getId()==info.getId())close();refresh();});}
    };
    if(manager!=null)manager.registerDeviceCallback(listener,handler);
  }
  private void refresh(){
    if(manager==null||spinner==null)return;
    List<MidiDeviceInfo> list=new ArrayList<>();List<String> names=new ArrayList<>();int selected=0;
    for(MidiDeviceInfo info:manager.getDevices()){
      if(info.getInputPortCount()<1)continue;
      list.add(info);String n=info.getProperties().getString(MidiDeviceInfo.PROPERTY_NAME);
      if(n==null)n="MIDI #"+info.getId();
      names.add(n+" · input ports "+info.getInputPortCount());
      if(n.toLowerCase(Locale.ROOT).contains("pa600")||n.toLowerCase(Locale.ROOT).contains("korg"))selected=list.size()-1;
    }
    choices=list.toArray(new MidiDeviceInfo[0]);
    if(names.isEmpty())names.add("لا يوجد USB MIDI");ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,names);spinner.setAdapter(adapter);spinner.setSelection(selected);
    if(status!=null&&choices.length==0&&port==null)status.setText("لم يظهر جهاز MIDI — افحص OTG");
  }
  private void close(){
    try{if(port!=null)port.close();}catch(Exception e){}
    try{if(device!=null)device.close();}catch(Exception e){}
    port=null;device=null;
  }
  private void connect(){
    int n=spinner.getSelectedItemPosition();
    if(n<0||n>=choices.length){status.setText("لا يوجد جهاز MIDI");return;}
    close();status.setText("جارٍ الاتصال...");
    MidiDeviceInfo info=choices[n];
    manager.openDevice(info,d->{
      if(d==null){status.setText("تعذّر فتح الجهاز");return;}
      MidiInputPort p=d.openInputPort(0);
      if(p==null){try{d.close();}catch(Exception e){}status.setText("تعذّر فتح MIDI input 0");return;}
      device=d;port=p;status.setText("متصل: "+info.getProperties().getString(MidiDeviceInfo.PROPERTY_NAME));
      record("PORT OPEN 0");
    },handler);
  }
  private void record(String text){
    messages.add(0,text);if(messages.size()>8)messages.remove(messages.size()-1);
    StringBuilder sb=new StringBuilder();for(String x:messages)sb.append(x).append("\n");if(log!=null)log.setText(sb.toString());
  }
  private void send(byte[] bytes,String name){
    if(port==null){status.setText("اربط Pa600 واضغط اتصال أولاً");return;}
    try{port.send(bytes,0,bytes.length);StringBuilder h=new StringBuilder();for(byte b:bytes)h.append(String.format(Locale.ROOT,"%02X ",b&255));record(name+" → "+h.toString());}
    catch(IOException e){status.setText("خطأ MIDI: "+e.getMessage());}
  }
  private EditText field(LinearLayout parent,String title,int initial){
    parent.addView(label(title,12,SOFT));
    EditText edit=new EditText(this);
    edit.setText(String.valueOf(initial));edit.setInputType(InputType.TYPE_CLASS_NUMBER);
    edit.setTextColor(TXT);edit.setTextSize(16);edit.setSingleLine(true);
    parent.addView(edit,new LinearLayout.LayoutParams(-1,dp(56)));
    return edit;
  }
  private int get7(EditText edit){
    int value=Integer.parseInt(edit.getText().toString().trim());
    if(value<0||value>127)throw new IllegalArgumentException("MIDI value must be 0–127");
    return value;
  }
  private void sendStyle(){
    try{
      int msb=get7(styleMsb),lsb=get7(styleLsb),pc=get7(stylePc);
      send(new byte[]{(byte)0xBF,0,(byte)msb},"STYLE CC0");
      send(new byte[]{(byte)0xBF,32,(byte)lsb},"STYLE CC32");
      sendPc(pc);
    }catch(Exception ex){status.setText("خطأ الأرقام: "+ex.getMessage());}
  }
  private void saveFavorite(){
    try{
      int msb=get7(styleMsb),lsb=get7(styleLsb),pc=get7(stylePc);
      JSONArray saved=new JSONArray(prefs.getString("styles","[]"));
      JSONObject o=new JSONObject();
      String name=styleLabel.getText().toString().trim();
      if(name.isEmpty())name="STYLE "+msb+"/"+lsb+"/"+pc;
      o.put("name",name);o.put("msb",msb);o.put("lsb",lsb);o.put("pc",pc);
      saved.put(o);prefs.edit().putString("styles",saved.toString()).apply();
      status.setText("تم حفظ "+name);
      refreshFavorites();
    }catch(Exception ex){status.setText("فشل الحفظ: "+ex.getMessage());}
  }
  private void refreshFavorites(){
    if(favoritesList==null)return;
    favoritesList.removeAllViews();
    try{
      JSONArray all=new JSONArray(prefs.getString("styles","[]"));
      if(all.length()>0){space(favoritesList,12);favoritesList.addView(label("المفضلة · اضغط للاستدعاء · اضغط مطولاً للحذف",12,SOFT));}
      for(int i=0;i<all.length();i++){
        JSONObject item=all.getJSONObject(i);
        final int index=i;
        final int msb=item.optInt("msb"),lsb=item.optInt("lsb"),pc=item.optInt("pc");
        Button b=button(item.optString("name","STYLE"),TILE,()->{
          send(new byte[]{(byte)0xBF,0,(byte)msb},"STYLE CC0");
          send(new byte[]{(byte)0xBF,32,(byte)lsb},"STYLE CC32");
          sendPc(pc);
        });
        b.setOnLongClickListener(v->{
          try{
            JSONArray previous=new JSONArray(prefs.getString("styles","[]"));
            JSONArray next=new JSONArray();
            for(int k=0;k<previous.length();k++)if(k!=index)next.put(previous.get(k));
            prefs.edit().putString("styles",next.toString()).apply();refreshFavorites();
          }catch(Exception ignored){}
          return true;
        });
        LinearLayout.LayoutParams p=new LinearLayout.LayoutParams(-1,dp(65));p.topMargin=dp(6);favoritesList.addView(b,p);
      }
    }catch(Exception ex){status.setText("خطأ قراءة المفضلة");}
  }
  private void sendPc(int program){send(new byte[]{(byte)0xCF,(byte)program},"PC "+program);}
  private void testNote(){
    if(port==null){status.setText("اتصل أولاً");return;}
    send(new byte[]{(byte)0x90,60,90},"NOTE ON Ch1");
    handler.postDelayed(()->send(new byte[]{(byte)0x80,60,0},"NOTE OFF Ch1"),350);
  }
  @Override protected void onDestroy(){if(manager!=null&&listener!=null)manager.unregisterDeviceCallback(listener);close();handler.removeCallbacksAndMessages(null);super.onDestroy();}
}
