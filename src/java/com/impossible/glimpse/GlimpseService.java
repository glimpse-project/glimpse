
package com.impossible.glimpse;

import android.widget.Toast;
import android.app.Service;
import android.os.IBinder;
import android.os.Messenger;
import android.os.Message;
import android.os.Handler;
import android.content.Intent;

import android.util.Log;

// ref: https://developer.android.com/guide/components/bound-services.html

public class GlimpseService extends Service {

    /** Command to the service to display a message */
    static final int MSG_SAY_HELLO = 1;


    /**
     * Handler of incoming messages from clients.
     */
    class IncomingHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what) {
                case MSG_SAY_HELLO:
                    Toast.makeText(getApplicationContext(), "hello!", Toast.LENGTH_SHORT).show();
                    break;
                default:
                    super.handleMessage(msg);
            }
        }
    }


    /**
     * Target we publish for clients to send messages to IncomingHandler.
     */
    final Messenger mMessenger = new Messenger(new IncomingHandler());


    @Override
    public void onCreate() {
        // The service is being created
        Log.d("GlimpseService", "onCreate called");
    }

    /**
     * When binding to the service, we return an interface to our messenger
     * for sending messages to the service.
     */
    @Override
    public IBinder onBind(Intent intent) {
        Log.d("GlimpseService", "onBind called");
        Toast.makeText(getApplicationContext(), "binding", Toast.LENGTH_SHORT).show();
        return mMessenger.getBinder();
    }

    @Override
    public boolean onUnbind(Intent intent) {
        // All clients have unbound with unbindService()
        return false;
    }


  @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d("GlimpseService", "onStartService called");
        // The service is starting, due to a call to startService()
        return START_STICKY;
    }
}
