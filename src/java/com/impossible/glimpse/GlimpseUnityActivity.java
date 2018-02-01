// Based on example from:
// https://docs.unity3d.com/Manual/AndroidUnityPlayerActivity.html
//
package com.impossible.glimpse;

import com.unity3d.player.UnityPlayerActivity;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.Context;
import android.content.ComponentName;

import com.impossible.glimpse.GlimpseJNI;
import com.impossible.glimpse.GlimpseConfig;


public class GlimpseUnityActivity extends UnityPlayerActivity
{
    ServiceConnection mTangoServiceConnection;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        if (GlimpseConfig.USE_TANGO) {
            Log.d("GlimpseUnityActivity", "onCreate called! (With Tango support)");

            // XXX: Make sure we try and load libtango_client_api.so before
            // the native activity tries to load our library which is linked
            // against libtango_client_api.so
            if (GlimpseJNI.loadTangoSharedLibrary() == GlimpseJNI.ARCH_ERROR) {
                Log.e("TangoJNINative", "ERROR! Unable to load libtango_client_api.so!");
            }

            System.loadLibrary("glimpse-unity-plugin");
        } else {
            Log.d("GlimpseUnityActivity", "onCreate called! (No Tango support)");
        }

        // call UnityPlayerActivity.onCreate()
        super.onCreate(savedInstanceState);

        if (GlimpseConfig.USE_TANGO) {
            mTangoServiceConnection = new ServiceConnection() {
                public void onServiceConnected(ComponentName name, IBinder service) {
                    GlimpseJNI.onTangoServiceConnected(service);
                    //setAndroidOrientation();
                    Log.d("GlimpseUnityActivity", "Tango Service Connected!");
                }
                public void onServiceDisconnected(ComponentName name) {
                    // Handle this if you need to gracefully shutdown/retry
                    // in the event that Tango itself crashes/gets upgraded while running.
                }
            };
        }
    }

    public static final boolean bindTangoService(final Context context,
            ServiceConnection connection)
    {
        Intent intent = new Intent();
        intent.setClassName("com.google.tango", "com.google.atap.tango.TangoService");

        if (context.getPackageManager().resolveService(intent, 0) != null)
        {
            return context.bindService(intent, connection, Context.BIND_AUTO_CREATE);
        }
        else
        {
            // TODO: bubble something up to the user!
            return false;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (GlimpseConfig.USE_TANGO) {
            bindTangoService(this, mTangoServiceConnection);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (GlimpseConfig.USE_TANGO) {
            //TangoJNINative.onPause();
            unbindService(mTangoServiceConnection);
        }
    }
}
