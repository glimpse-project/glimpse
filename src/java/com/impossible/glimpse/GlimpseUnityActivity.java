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

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.view.OrientationEventListener;
import android.view.Surface;

import com.impossible.glimpse.GlimpseConfig;
import com.impossible.glimpse.GlimpseJNI;
import com.unity3d.player.UnityPlayerActivity;


public class GlimpseUnityActivity extends UnityPlayerActivity
{
    private ServiceConnection mTangoServiceConnection;
    private OrientationEventListener mOrientationListener;

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

        } else {
            Log.d("GlimpseUnityActivity", "onCreate called! (No Tango support)");
        }

        System.loadLibrary("glimpse-unity-plugin");
        // call UnityPlayerActivity.onCreate()
        super.onCreate(savedInstanceState);
        mOrientationListener = new MyOrientationEventListener(this);

        if (GlimpseConfig.USE_TANGO) {
            mTangoServiceConnection = new ServiceConnection() {
                public void onServiceConnected(ComponentName name, IBinder service) {
                    GlimpseJNI.onTangoServiceConnected(service);
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

        if (mOrientationListener.canDetectOrientation()) {
            mOrientationListener.enable();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (GlimpseConfig.USE_TANGO) {
            //TangoJNINative.onPause();
            unbindService(mTangoServiceConnection);
        }

        mOrientationListener.disable();
    }

    native static void OnDisplayRotate(int rotation);

    private class MyOrientationEventListener extends OrientationEventListener {

        private static final float ALPHA = 0.4f;
        int rotation;
        int orientation;

        MyOrientationEventListener(Context context) {
            super(context);
        }

        @Override
        public void onOrientationChanged(int newOrientation) {

            if (newOrientation == ORIENTATION_UNKNOWN) {
                return;
            }

            orientation = filter(newOrientation, orientation);
            int newRotation;

            if (orientation >= 60 && orientation <= 140) {
                newRotation = Surface.ROTATION_270;
            } else if (orientation >= 140 && orientation <= 220) {
                newRotation = Surface.ROTATION_180;
            } else if (orientation >= 220 && orientation <= 300) {
                newRotation = Surface.ROTATION_90;
            } else {
                newRotation = Surface.ROTATION_0;
            }

            if (rotation != newRotation) {
                rotation = newRotation;
                Log.d("GlimpseUnityActivity", "calling OnDisplayRotation " + rotation);
                OnDisplayRotate(rotation);
            }

        }

        private double linearInterpolation(double x0, double x1, float alpha) {
            return x0 * (1.0f - alpha) + x1 * alpha;
        }

        private int filter(int current, int previous) {
            double radPrevious = Math.toRadians(previous);
            double radCurrent = Math.toRadians(current);
            double sin = linearInterpolation(Math.sin(radPrevious),
                    Math.sin(radCurrent), ALPHA);
            double cos = linearInterpolation(Math.cos(radPrevious),
                    Math.cos(radCurrent), ALPHA);
            double radian = Math.atan2(sin, cos);
            double degrees = Math.toDegrees(radian);
            return (int) (degrees < 0 ? degrees + 360 : degrees);
        }
    }
}
