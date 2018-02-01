/*
 * Copyright (C) 2018 Glimp IP Ltd
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.impossible.glimpse;

import android.app.NativeActivity;
import android.content.pm.PackageManager;
import android.Manifest;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.Context;
import android.content.ComponentName;

import com.impossible.glimpse.GlimpseJNI;
import com.impossible.glimpse.GlimpseConfig;

public class GlimpseNativeActivity extends NativeActivity
{
    ServiceConnection mTangoServiceConnection;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        if (GlimpseConfig.USE_TANGO) {
            // XXX: Make sure we try and load libtango_client_api.so before
            // the native activity tries to load our library which is linked
            // against libtango_client_api.so
            if (GlimpseJNI.loadTangoSharedLibrary() == GlimpseJNI.ARCH_ERROR) {
                Log.e("TangoJNINative", "ERROR! Unable to load libtango_client_api.so!");
            }
        }

        // FIXME: avoid having this class be glimpse_viewer specific...
        System.loadLibrary("glimpse_viewer_android");

        super.onCreate(savedInstanceState);

        Log.d("GlimpseUnityActivity", "onCreate called!");

        String[] accessPermissions = new String[] {
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        requestPermissions(accessPermissions, 1);

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

    @Override
    public void onRequestPermissionsResult(int requestCode,
            String[] permissions,
            int[] grantResults)
    {
        if (requestCode != 1) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }
        Log.d("GlimpseNativeActivity", "onRequestPermissionsResult called!");

        OnPermissionsCheckResult(
                grantResults[0] == PackageManager.PERMISSION_GRANTED &&
                grantResults[1] == PackageManager.PERMISSION_GRANTED &&
                grantResults[2] == PackageManager.PERMISSION_GRANTED);
    }

    native static void OnPermissionsCheckResult(boolean granted);
}
