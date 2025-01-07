import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { VideodownloadComponent } from './videodownload/videodownload.component'; // Importez votre composant ici
import { VideouploadComponent } from './videoupload/videoupload.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from './auth.guard';

import { AppComponent } from './app.component';
const routes: Routes = [
  { path: '', component: LoginComponent },
  { path: 'videodownload', component: VideodownloadComponent, canActivate: [AuthGuard] },
  { path: 'videoupload', component: VideouploadComponent, canActivate: [AuthGuard] },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
